"""Character/word error rates without case, accent or punctuation normalization."""
import argparse
import json
from pathlib import Path


def distance(a, b):
    prev = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row = [i]
        for j,y in enumerate(b,1): row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev = row
    return prev[-1]


def transcription_metrics(references, predictions):
    if not references or len(references)!=len(predictions): raise ValueError('Pares de referências e transcrições são obrigatórios.')
    chars = sum(len(s) for s in references); words = sum(len(s.split()) for s in references)
    if not chars or not words: raise ValueError('Referências vazias não permitem CER/WER.')
    return {'samples':len(references), 'cer':sum(distance(a,b) for a,b in zip(references,predictions))/chars, 'wer':sum(distance(a.split(),b.split()) for a,b in zip(references,predictions))/words}


def main():
    p=argparse.ArgumentParser(description='Avaliação real OCR com imagens e transcrições humanas JSONL.')
    p.add_argument('manifest',type=Path); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from app.core.config import Settings
    from app.services.ocr_service import OCRService
    service=OCRService(Settings.from_env()); refs=[]; predictions=[]; details=[]
    for line in args.manifest.read_text().splitlines():
        item=json.loads(line); image=(args.manifest.parent/item['image']).resolve()
        result=service.transcribe([image]); refs.append(item['reference']); predictions.append(result['text'])
        details.append({'image':item['image'],'warnings':result['warnings'],'prediction':result['text']})
    output={'metrics':transcription_metrics(refs,predictions),'samples':details,'model':service.status()}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
