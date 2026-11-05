from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptionUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=30000)
    confirmed: bool = False
    theme: str | None = Field(default=None, max_length=2000)
    motivating_text: str | None = Field(default=None, max_length=20000)

    @field_validator('text')
    @classmethod
    def not_blank(cls, value):
        if not value.strip(): raise ValueError('A transcrição não pode estar vazia.')
        return value


class HumanReferenceInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scores: dict[str, int]
    notes: str = Field(default='', max_length=10000)
    consent_training: bool = False
    partition: Literal['validation','train','test'] = 'validation'

    @field_validator('scores', mode='before')
    @classmethod
    def valid_scores(cls,value):
        if not isinstance(value,dict) or set(value) != {'c1','c2','c3','c4','c5'}:
            raise ValueError('Informe as cinco competências.')
        if any(type(v) is not int or v not in (0,40,80,120,160,200) for v in value.values()):
            raise ValueError('Notas devem ser inteiros de 0 a 200, em intervalos de 40.')
        return value
