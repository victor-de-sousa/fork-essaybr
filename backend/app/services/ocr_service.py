"""Local TrOCR inference. Line segmentation is experimental for Portuguese manuscripts."""
from pathlib import Path
from threading import Lock
import numpy as np
from PIL import Image, ImageOps


def segment_lines(image: Image.Image):
    gray = np.asarray(ImageOps.autocontrast(image.convert('L')))
    ink = gray < min(210, float(np.percentile(gray, 50))*0.82)
    active = ink.sum(axis=1) >= max(3, image.width*0.003)
    bands=[]; start=None; last=None
    for y, used in enumerate(active):
        if used:
            if start is None: start=y
            last=y
        elif start is not None and y-last > 3:
            if last-start>=2: bands.append((start,last+1))
            start=None
    if start is not None and last-start>=2: bands.append((start,last+1))
    result=[]
    for top,bottom in bands:
        columns=np.flatnonzero(ink[top:bottom].any(axis=0))
        if not len(columns): continue
        box=(max(0,int(columns[0])-8),max(0,top-5),min(image.width,int(columns[-1])+9),min(image.height,bottom+5))
        result.append((image.crop(box),top,bottom))
    return result


class OCRService:
    def __init__(self, settings):
        self.settings=settings; self.processor=None; self.model=None; self.lock=Lock()

    def status(self):
        available=(self.settings.ocr_model_dir/'config.json').is_file() and any((self.settings.ocr_model_dir/name).is_file() for name in ['model.safetensors','pytorch_model.bin','model.safetensors.index.json'])
        return {'available':available,'loaded':self.model is not None,'provider':'TrOCR local por linhas','portuguese_validated':False,'message':'Modelo OCR local ausente. Execute scripts/download_models.py --ocr.' if not available else 'Valide CER/WER com manuscritos portugueses antes de confiar na transcrição.'}

    def _load(self):
        if not self.status()['available']: raise RuntimeError('Modelo OCR local ausente. Baixe os pesos com scripts/download_models.py --ocr.')
        if self.model is not None: return
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.device = 'cuda' if torch.cuda.is_available() and self.settings.device=='auto' else ('cpu' if self.settings.device=='auto' else self.settings.device)
            self.processor=TrOCRProcessor.from_pretrained(str(self.settings.ocr_model_dir),local_files_only=True)
            self.model=VisionEncoderDecoderModel.from_pretrained(str(self.settings.ocr_model_dir),local_files_only=True).to(self.device).eval()
        except (ImportError,OSError) as exc: raise RuntimeError('OCR indisponível: instale requirements-ml e verifique os pesos locais completos.') from exc

    def transcribe(self, paths: list[Path]):
        with self.lock:
            self._load()
            import torch
            pages=[]; warnings=['OCR experimental: confirme acentos, pontuação, parágrafos e todos os trechos antes de avaliar. Não há correção ortográfica automática.']
            for page_number,path in enumerate(paths,1):
                with Image.open(path) as source: lines=segment_lines(source.convert('RGB'))
                if len(lines)>100: raise ValueError('Segmentação encontrou mais de 100 linhas; recorte/alinhe a página e tente novamente.')
                heights=[bottom-top for _,top,bottom in lines]; typical=float(np.median(heights)) if heights else 1
                text=''; line_details=[]; last_bottom=None
                for line_number,(line,top,bottom) in enumerate(lines,1):
                    inputs=self.processor(images=line,return_tensors='pt').pixel_values.to(self.device)
                    with torch.inference_mode():
                        output=self.model.generate(inputs,max_new_tokens=256,return_dict_in_generate=True,output_scores=True)
                        transitions=self.model.compute_transition_scores(output.sequences,output.scores,normalize_logits=True)
                    confidence=float(transitions.exp().mean().item()) if transitions.numel() else 0.0
                    recognized=self.processor.batch_decode(output.sequences,skip_special_tokens=True)[0]
                    uncertain=confidence<self.settings.ocr_min_confidence or len(output.scores)>=256 or not recognized.strip()
                    if uncertain:
                        recognized='[ilegível]'; warnings.append(f'Página {page_number}, linha {line_number}: leitura incerta; confira a imagem.')
                    separator='\n\n' if last_bottom is not None and top-last_bottom>typical*1.4 else ' '
                    text+=(separator if text else '')+recognized
                    line_details.append({'line':line_number,'y_start':top,'y_end':bottom,'confidence':confidence,'uncertain':uncertain,'text':recognized})
                    last_bottom=bottom
                if not lines:
                    text='[ilegível]'; warnings.append(f'Página {page_number}: nenhuma linha reconhecível.')
                pages.append({'page':page_number,'text':text,'lines':line_details})
            return {'text':'\n\n'.join(p['text'] for p in pages),'warnings':warnings,'pages':pages}
