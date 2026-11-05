"""Loopback-first API. Launch with: PYTHONPATH=backend python -P -m uvicorn app.main:app."""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import select
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import Settings
from app.database.store import (AssessmentAttempt, CompetencyScore, Correction, Essay, EssayImage,
                                HumanReference, Issue, Transcription, connect)
from app.schemas.essays import HumanReferenceInput, TranscriptionUpdate
from app.services.image_service import validate_image
from app.services.ocr_service import OCRService


def create_app(settings=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        yield
        app.state.engine.dispose()

    app=FastAPI(title='Caderno ENEM — correção local',version='0.1.0',lifespan=lifespan)
    settings.images_dir.mkdir(parents=True,exist_ok=True)
    engine,sessions=connect(settings.database_path)
    app.state.settings=settings; app.state.engine=engine; app.state.sessions=sessions
    app.state.ocr=OCRService(settings)
    app.state.write_lock=RLock()
    # Single local worker; serialize writes/inference so transcript changes cannot race grading.
    app.add_middleware(CORSMiddleware,allow_origins=list(settings.cors_origins),allow_methods=['GET','POST','PATCH','PUT','DELETE'],allow_headers=['Content-Type'])
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request,exc):
        return JSONResponse({'error':{'code':f'http_{exc.status_code}','message':str(exc.detail)}},status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        return JSONResponse({'error':{'code':'invalid_request','message':'Dados inválidos: confira os campos e as notas informadas.','fields':[{'field':'.'.join(map(str,e['loc'])),'message':e['msg']} for e in exc.errors()]}},status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request,exc):
        return JSONResponse({'error':{'code':'internal_error','message':'Falha interna. Verifique a instalação e tente novamente.'}},status_code=500)

    @app.middleware('http')
    async def local_request_guard(request,call_next):
        origin=request.headers.get('origin')
        if request.method not in ('GET','HEAD','OPTIONS') and origin and origin not in settings.cors_origins:
            return JSONResponse({'error':{'code':'origin_denied','message':'Origem não permitida.'}},status_code=403)
        length=request.headers.get('content-length')
        maximum=(settings.max_pages*settings.max_image_mb+2)*1024*1024
        if length:
            try: too_large=int(length)>maximum
            except ValueError: too_large=True
            if too_large: return JSONResponse({'error':{'code':'upload_too_large','message':'Envio excede o limite de tamanho.'}},status_code=413)
        return await call_next(request)

    def find(db,essay_id):
        try: uuid.UUID(essay_id)
        except ValueError: raise HTTPException(404,'Redação não encontrada.')
        essay=db.get(Essay,essay_id)
        if essay is None: raise HTTPException(404,'Redação não encontrada.')
        return essay

    def view(essay):
        current=next((c for c in essay.corrections if c.id==essay.current_correction_id),None)
        return {'id':essay.id,'theme':essay.theme,'motivating_text':essay.motivating_text,'status':essay.status,
                'created_at':essay.created_at.isoformat(),'transcription':essay.transcription,'confirmed':essay.confirmed,
                'ocr_warnings':essay.ocr_warnings,'error':essay.error,
                'images':[{'id':i.id,'position':i.position,'url':f'/api/essays/{essay.id}/images/{i.id}'} for i in essay.images],
                'correction':current.payload if current else None}

    def rubric_service():
        from app.services.rubric_service import RubricService
        return RubricService(settings)

    def correction_service():
        from app.services.correction_service import CorrectionService
        return getattr(app.state,'corrector',None) or CorrectionService(settings)

    def predictor():
        from app.ml.predict import Predictor
        if not hasattr(app.state,'predictor'): app.state.predictor=Predictor(settings.model_dir)
        return app.state.predictor

    def fail_operation(db,essay,exc):
        essay.status='failed'; essay.error=str(exc); essay.current_correction_id=None
        audit=getattr(exc,'audit',None)
        essay.attempts.append(AssessmentAttempt(status='failed',payload={'error':str(exc),'audit':audit,'transcription':essay.transcription}))
        db.commit()
        raise HTTPException(422 if isinstance(exc,ValueError) else 503,str(exc))

    @app.get('/api/health')
    def health(): return {'status':'ok','local_only':True}

    @app.post('/api/essays/upload',status_code=201)
    def upload(files:list[UploadFile]=File(...),theme:str=Form(''),motivating_text:str=Form('')):
        if not 1<=len(files)<=settings.max_pages: raise HTTPException(422,f'Envie entre 1 e {settings.max_pages} páginas.')
        if len(theme)>2000 or len(motivating_text)>20000: raise HTTPException(422,'Tema ou textos motivadores excedem o limite.')
        validated=[]
        try:
            for file in files:
                raw=file.file.read(settings.max_image_mb*1024*1024+1)
                validated.append(validate_image(raw,file.filename or '',file.content_type or '',settings))
        except ValueError as exc: raise HTTPException(422,str(exc)) from exc
        finally:
            for file in files: file.file.close()
        essay_id=str(uuid.uuid4()); folder=settings.images_dir/essay_id
        with app.state.write_lock,sessions() as db:
            try:
                folder.mkdir()
                essay=Essay(id=essay_id,theme=theme,motivating_text=motivating_text)
                for position,(data,suffix) in enumerate(validated,1):
                    name=f'page_{position:02d}{suffix}'; (folder/name).write_bytes(data)
                    essay.images.append(EssayImage(id=str(uuid.uuid4()),position=position,relative_path=f'{essay_id}/{name}'))
                db.add(essay); db.commit()
                return view(essay)
            except Exception:
                db.rollback(); shutil.rmtree(folder,ignore_errors=True); raise

    @app.get('/api/essays')
    def history():
        with sessions() as db: return [view(e) for e in db.scalars(select(Essay).order_by(Essay.created_at.desc())).all()]

    @app.get('/api/essays/{essay_id}')
    def get_essay(essay_id:str):
        with sessions() as db: return view(find(db,essay_id))

    @app.get('/api/essays/{essay_id}/images/{image_id}')
    def image(essay_id:str,image_id:str):
        with sessions() as db:
            essay=find(db,essay_id); item=next((i for i in essay.images if i.id==image_id),None)
            if item is None: raise HTTPException(404,'Página não encontrada.')
            path=(settings.images_dir/item.relative_path).resolve()
            if not path.is_relative_to(settings.images_dir.resolve()) or not path.is_file(): raise HTTPException(404,'Imagem não encontrada.')
            return FileResponse(path,media_type='image/png',headers={'Cache-Control':'no-store'})

    @app.post('/api/essays/{essay_id}/transcribe')
    def transcribe(essay_id:str):
        with app.state.write_lock,sessions() as db:
            essay=find(db,essay_id)
            try: result=app.state.ocr.transcribe([settings.images_dir/i.relative_path for i in essay.images])
            except (ValueError,RuntimeError,OSError) as exc: fail_operation(db,essay,exc)
            essay.transcription=result['text']; essay.confirmed=False; essay.ocr_warnings=result['warnings']
            essay.status='transcribed'; essay.error=None; essay.current_correction_id=None
            essay.transcriptions.append(Transcription(text=result['text'],confirmed=False,source='ocr'))
            essay.attempts.append(AssessmentAttempt(status='ocr',payload=result))
            db.commit(); return view(essay)

    @app.patch('/api/essays/{essay_id}/transcription')
    def update_transcription(essay_id:str,body:TranscriptionUpdate):
        with app.state.write_lock,sessions() as db:
            essay=find(db,essay_id)
            essay.transcription=body.text; essay.confirmed=body.confirmed
            if body.theme is not None: essay.theme=body.theme
            if body.motivating_text is not None: essay.motivating_text=body.motivating_text
            essay.current_correction_id=None; essay.error=None; essay.status='confirmed' if body.confirmed else 'transcribed'
            essay.transcriptions.append(Transcription(text=body.text,confirmed=body.confirmed,source='human_review'))
            db.commit(); return view(essay)

    @app.post('/api/essays/{essay_id}/correct')
    def correct(essay_id:str):
        with app.state.write_lock,sessions() as db:
            essay=find(db,essay_id)
            if not essay.confirmed or not essay.transcription.strip(): raise HTTPException(409,'Confira e confirme a transcrição antes da correção.')
            if not essay.theme.strip(): raise HTTPException(422,'Informe o tema para avaliar a competência 2.')
            if '[ilegível]' in essay.transcription.lower(): raise HTTPException(422,'Resolva os trechos ilegíveis antes da avaliação definitiva.')
            try:
                prediction=predictor().predict(essay.transcription)
                result=correction_service().correct(essay.transcription,essay.theme,essay.motivating_text,prediction)
                scores=result['scores']
                # Defense in depth: service schema validates evidence; API validates the persisted score shape.
                HumanReferenceInput(scores=scores)
                if result['total_score']!=sum(scores.values()): raise ValueError('Total de notas inconsistente.')
            except (ValueError,RuntimeError,OSError) as exc: fail_operation(db,essay,exc)
            correction_id=str(uuid.uuid4())
            payload={**result,'id':correction_id,'essay_id':essay.id,'theme':essay.theme,'transcription':essay.transcription,'status':'completed','model_info':prediction['model_info']}
            record=Correction(id=correction_id,payload=payload,model_metadata=prediction['model_info'])
            record.scores=[CompetencyScore(competency=k,score=v) for k,v in scores.items()]
            for c in result.get('competencies',[]):
                for issue in c.get('issues',[]): record.issues.append(Issue(competency=str(c['id']),payload=issue))
            essay.corrections.append(record); essay.current_correction_id=correction_id
            essay.status='completed'; essay.error=None
            db.commit(); return payload

    @app.get('/api/essays/{essay_id}/correction')
    def get_correction(essay_id:str):
        with sessions() as db:
            result=view(find(db,essay_id))['correction']
            if result is None: raise HTTPException(404,'Esta redação ainda não tem uma correção vigente.')
            return result

    @app.get('/api/essays/{essay_id}/audit')
    def audit(essay_id:str):
        with sessions() as db:
            essay=find(db,essay_id)
            return {'corrections':[c.payload for c in essay.corrections], 'attempts':[{'status':a.status,'created_at':a.created_at.isoformat(),'payload':a.payload} for a in essay.attempts],
                    'transcriptions':[{'text':t.text,'confirmed':t.confirmed,'source':t.source,'created_at':t.created_at.isoformat()} for t in essay.transcriptions]}

    @app.delete('/api/essays/{essay_id}',status_code=204)
    def delete(essay_id:str):
        with app.state.write_lock,sessions() as db:
            essay=find(db,essay_id); folder=settings.images_dir/essay.id
            if folder.exists(): shutil.rmtree(folder)
            db.delete(essay); db.commit()
        return Response(status_code=204)

    @app.get('/api/model/status')
    def status():
        from app.services.feedback_service import OllamaClient
        try: scoring=predictor().status()
        except (ValueError,RuntimeError,OSError) as exc: scoring={'trained':False,'available':False,'message':str(exc)}
        try: rubric=rubric_service().status()
        except (ValueError,RuntimeError,OSError) as exc: rubric={'available':False,'message':str(exc)}
        # Probe via a fixed local URL; requests never transmit essays here.
        import httpx
        from urllib.parse import urlparse
        url=urlparse(settings.ollama_url)
        ollama={'available':False,'model':settings.ollama_model}
        if url.scheme=='http' and url.hostname in ('127.0.0.1','localhost','::1'):
            try:
                response=httpx.get(settings.ollama_url.rstrip('/')+'/api/tags',timeout=2,trust_env=False,follow_redirects=False)
                response.raise_for_status(); names=[m['name'] for m in response.json().get('models',[])]
                ollama['available']=settings.ollama_model in names
            except (httpx.HTTPError,ValueError,KeyError): pass
        return {'scoring':scoring,'ocr':app.state.ocr.status(),'ollama':ollama,'rubric':rubric}

    @app.post('/api/rubric/upload')
    def upload_rubric(file:UploadFile=File(...)):
        try:
            raw=file.file.read(30*1024*1024+1)
            if len(raw)>30*1024*1024: raise ValueError('A cartilha pode ter até 30 MB.')
            with app.state.write_lock: return rubric_service().ingest(raw,file.filename or '')
        except ValueError as exc: raise HTTPException(422,str(exc))
        except RuntimeError as exc: raise HTTPException(503,str(exc))
        finally: file.file.close()

    @app.get('/api/rubric')
    def get_rubric():
        try: return rubric_service().get_rubric()
        except (ValueError,RuntimeError) as exc: return {**rubric_service().status(),'message':str(exc)}

    @app.put('/api/rubric/approve')
    def approve_rubric(body:dict):
        try:
            with app.state.write_lock: return rubric_service().approve(body)
        except ValueError as exc: raise HTTPException(422,str(exc))
        except RuntimeError as exc: raise HTTPException(503,str(exc))

    @app.post('/api/essays/{essay_id}/reference',status_code=201)
    def human_reference(essay_id:str,body:HumanReferenceInput):
        if body.partition=='train' and not body.consent_training: raise HTTPException(422,'Uso para treinamento exige consentimento explícito.')
        with app.state.write_lock,sessions() as db:
            essay=find(db,essay_id)
            if not essay.current_correction_id: raise HTTPException(409,'Conclua uma correção antes de inserir a referência humana.')
            previous={ref.partition for ref in essay.references}
            if previous and body.partition not in previous: raise HTTPException(409,'Uma mesma redação não pode ser reutilizada entre treino, validação e teste.')
            item=HumanReference(correction_id=essay.current_correction_id,**body.model_dump())
            essay.references.append(item); db.commit()
            ai=view(essay)['correction']['scores']
            return {'id':item.id,'scores':item.scores,'ai_scores':ai,'differences':{k:ai[k]-item.scores[k] for k in ai},'total_difference':sum(ai.values())-sum(item.scores.values()),'consent_training':item.consent_training,'partition':item.partition}

    @app.get('/api/validation/report')
    def validation_report():
        from app.ml.metrics import score_metrics
        keys=['c1','c2','c3','c4','c5']; rows=[]
        with sessions() as db:
            # Most recent teacher reference per correction prevents repeat submissions inflating n.
            latest={}
            for ref in db.scalars(select(HumanReference).order_by(HumanReference.created_at)).all(): latest[ref.correction_id]=ref
            for ref in latest.values():
                correction=db.get(Correction,ref.correction_id)
                if correction:
                    ai=correction.payload['scores']
                    rows.append({'essay_id':ref.essay_id,'reference_id':ref.id,'human_scores':ref.scores,'ai_scores':ai,'differences':{k:ai[k]-ref.scores[k] for k in keys},'total_difference':sum(ai.values())-sum(ref.scores.values()),'partition':ref.partition,'model_version':correction.model_metadata.get('version'),'rubric_version':correction.payload.get('rubric_info',{}).get('version')})
        groups={}
        for row in rows:
            key=f"{row['model_version']}|{row['rubric_version']}|{row['partition']}"; groups.setdefault(key,[]).append(row)
        return {'count':len(rows),'comparisons':rows,'groups':{key:{'count':len(group),'metrics':score_metrics([[r['human_scores'][k] for k in keys] for r in group],[[r['ai_scores'][k] for k in keys] for r in group])} for key,group in groups.items()},'message':'Referências são apenas avaliação; nenhum treinamento automático é realizado.'}

    return app


app=create_app()
