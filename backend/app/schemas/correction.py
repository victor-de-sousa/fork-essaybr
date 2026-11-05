"""Closed structured contracts for reviewed norms and grounded evaluations."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator


Score = Literal[0, 40, 80, 120, 160, 200]
CompetencyId = Literal["c1", "c2", "c3", "c4", "c5"]
NonEmpty = Annotated[str, Field(min_length=1, max_length=12000)]
COMPETENCY_IDS = {"c1", "c2", "c3", "c4", "c5"}
SCORES = {0, 40, 80, 120, 160, 200}


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(ClosedModel):
    page: Annotated[StrictInt, Field(ge=1)]
    excerpt: NonEmpty

    @field_validator("excerpt")
    @classmethod
    def meaningful_excerpt(cls, value):
        if len(value.strip()) < 12:
            raise ValueError("A citação deve conter um trecho verificável da fonte.")
        return value


class Descriptor(ClosedModel):
    score: Score
    criteria_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{2,80}$")]
    description: NonEmpty
    citation: Citation

    @field_validator("score", mode="before")
    @classmethod
    def strict_score(cls, value):
        if type(value) is not int:
            raise ValueError("A nota deve ser um inteiro da escala ENEM.")
        return value


class CompetencyRubric(ClosedModel):
    id: CompetencyId
    name: NonEmpty
    descriptors: Annotated[list[Descriptor], Field(min_length=6, max_length=6)]

    @model_validator(mode="after")
    def six_distinct_levels(self):
        if {d.score for d in self.descriptors} != SCORES:
            raise ValueError("Cada competência exige os seis descritores de 0 a 200.")
        return self


class SpecialRule(ClosedModel):
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{2,80}$")]
    description: NonEmpty
    fact: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{2,80}$")]
    action: Literal["zero_total", "zero_competency", "cap_competency", "informational"]
    competency_id: CompetencyId | None = None
    score: Score | None = None
    required_context: list[Literal["original_line_count", "motivating_text", "theme", "original_image_review"]] = Field(default_factory=list)
    citation: Citation

    @model_validator(mode="after")
    def valid_action(self):
        if self.action in {"zero_competency", "cap_competency"} and not self.competency_id:
            raise ValueError("Regra por competência exige competency_id.")
        if self.action == "cap_competency" and (self.score is None or type(self.score) is not int):
            raise ValueError("O teto da regra deve ser uma nota da escala ENEM.")
        if self.action != "cap_competency" and self.score is not None:
            raise ValueError("Somente regras de teto devem informar score.")
        return self


class OfficialExample(ClosedModel):
    text: NonEmpty
    citation: Citation


class Rubric(ClosedModel):
    year: Literal[2026]
    title: NonEmpty
    identity_citation: Citation
    competencies: Annotated[list[CompetencyRubric], Field(min_length=5, max_length=5)]
    special_rules: Annotated[list[SpecialRule], Field(min_length=1, max_length=100)]
    examples: Annotated[list[OfficialExample], Field(max_length=50)] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_matrix(self):
        if {c.id for c in self.competencies} != COMPETENCY_IDS:
            raise ValueError("A rubrica exige exatamente c1, c2, c3, c4 e c5.")
        criteria = [d.criteria_id for c in self.competencies for d in c.descriptors]
        if len(set(criteria)) != 30:
            raise ValueError("Os 30 identificadores de critério devem ser únicos.")
        if len({r.id for r in self.special_rules}) != len(self.special_rules):
            raise ValueError("Identificadores de regra duplicados.")
        return self


class RubricApproval(ClosedModel):
    document_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    official_source_confirmed: StrictBool
    special_rules_reviewed: StrictBool
    reviewer: Annotated[str, Field(min_length=2, max_length=200)]
    rubric: Rubric

    @model_validator(mode="after")
    def explicit_review(self):
        if not self.official_source_confirmed or not self.special_rules_reviewed or not self.reviewer.strip():
            raise ValueError("Confirme a origem oficial e a revisão completa das regras especiais.")
        return self


class Evidence(ClosedModel):
    text: NonEmpty
    start: Annotated[StrictInt, Field(ge=0)]
    end: Annotated[StrictInt, Field(gt=0)]

    @model_validator(mode="after")
    def valid_range(self):
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("Os limites da evidência não correspondem ao trecho.")
        return self


class Issue(ClosedModel):
    kind: NonEmpty
    explanation: NonEmpty
    evidence: Evidence
    suggestion: NonEmpty


class CompetencyEvaluation(ClosedModel):
    id: CompetencyId
    score: Score
    criteria_id: str
    justification: NonEmpty
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=30)]
    issues: Annotated[list[Issue], Field(max_length=40)]
    strengths: Annotated[list[NonEmpty], Field(max_length=20)]
    improvements: Annotated[list[NonEmpty], Field(max_length=20)]
    citation: Citation

    @field_validator("score", mode="before")
    @classmethod
    def strict_score(cls, value):
        if type(value) is not int:
            raise ValueError("Nota deve ser um inteiro da escala ENEM.")
        return value


class Fact(ClosedModel):
    id: str
    value: StrictBool | None
    explanation: NonEmpty
    evidence: Annotated[list[Evidence], Field(max_length=30)]


class Evaluation(ClosedModel):
    competencies: Annotated[list[CompetencyEvaluation], Field(min_length=5, max_length=5)]
    total_score: Annotated[StrictInt, Field(ge=0, le=1000)]
    general_feedback: NonEmpty
    facts: Annotated[list[Fact], Field(max_length=100)]

    @model_validator(mode="after")
    def complete_and_consistent(self):
        if {c.id for c in self.competencies} != COMPETENCY_IDS:
            raise ValueError("São obrigatórias exatamente as cinco competências.")
        if sum(c.score for c in self.competencies) != self.total_score:
            raise ValueError("A nota total não corresponde à soma das competências.")
        if len({f.id for f in self.facts}) != len(self.facts):
            raise ValueError("Fatos repetidos na avaliação.")
        return self


class Resolution(ClosedModel):
    id: str
    reason: NonEmpty


class Review(ClosedModel):
    evaluation: Evaluation
    resolutions: Annotated[list[Resolution], Field(min_length=1, max_length=105)]
