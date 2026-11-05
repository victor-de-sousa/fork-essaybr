"""Paths are operator configuration, never received from API clients."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    root_dir: Path = Path(__file__).resolve().parents[3]
    model_dir: Path | None = None
    ocr_model_dir: Path | None = None
    embedding_model_dir: Path | None = None
    rubric_dir: Path | None = None
    ollama_url: str = 'http://127.0.0.1:11434'
    ollama_model: str = 'qwen2.5:7b'
    device: str = 'auto'
    max_pages: int = 10
    max_image_mb: int = 10
    max_image_pixels: int = 25_000_000
    ocr_min_confidence: float = 0.70
    cors_origins: tuple[str, ...] = ('http://localhost:5173', 'http://127.0.0.1:5173')

    def __post_init__(self):
        self.root_dir = Path(self.root_dir).resolve()
        for key, default in [('model_dir','artifacts/models/baseline'), ('ocr_model_dir','artifacts/models/ocr'), ('embedding_model_dir','artifacts/models/embeddings'), ('rubric_dir','docs/enem')]:
            value = Path(getattr(self, key) or default)
            setattr(self,key,value if value.is_absolute() else self.root_dir / value)

    @property
    def images_dir(self): return self.root_dir / 'images'

    @property
    def database_path(self): return self.root_dir / 'artifacts' / 'corrections.sqlite3'

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(cls.root_dir / '.env')
        args = {}
        for key in ['model_dir','ocr_model_dir','embedding_model_dir','rubric_dir','ollama_url','ollama_model','device']:
            if key.upper() in os.environ: args[key] = os.environ[key.upper()]
        for key in ['max_pages','max_image_mb','max_image_pixels']:
            if key.upper() in os.environ: args[key] = int(os.environ[key.upper()])
        if 'OCR_MIN_CONFIDENCE' in os.environ: args['ocr_min_confidence'] = float(os.environ['OCR_MIN_CONFIDENCE'])
        if 'CORS_ORIGINS' in os.environ: args['cors_origins'] = tuple(x.strip() for x in os.environ['CORS_ORIGINS'].split(',') if x.strip())
        return cls(**args)
