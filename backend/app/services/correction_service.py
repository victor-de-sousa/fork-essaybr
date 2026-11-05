"""Grounded local assessments; no scores are returned without verified dependencies."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import uuid

from app.schemas.correction import COMPETENCY_IDS, SCORES, Evaluation, Review
from app.services.feedback_service import OllamaClient, SYSTEM_INSTRUCTIONS
from app.services.rubric_service import RubricService, normalized


DISCLAIMER = ("Estimativa automática de uso educacional. A consistência entre duas avaliações locais é experimental "
              "e não equivale à correção oficial do INEP nem comprova a validade da nota. Requer revisão humana.")


class CorrectionError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


class CorrectionUnavailable(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


class CorrectionService:
    def __init__(self, settings):
        self.settings = settings
        self.rubrics = RubricService(settings)
        self.llm = OllamaClient(settings)

    @staticmethod
    def _validate_prediction(prediction):
        if not isinstance(prediction, dict) or prediction.get("model_info", {}).get("trained") is not True:
            raise ValueError("É obrigatório um modelo supervisionado realmente treinado antes da correção.")
        scores = prediction.get("scores", {})
        if set(scores) != COMPETENCY_IDS or any(type(value) is not int or value not in SCORES for value in scores.values()):
            raise ValueError("Previsão supervisionada exige as cinco notas na escala ENEM.")
        if not prediction["model_info"].get("version"):
            raise ValueError("Modelo treinado sem identificação de versão.")

    @staticmethod
    def _validate_evidence(evidence, text):
        if evidence.end > len(text) or text[evidence.start:evidence.end] != evidence.text:
            raise ValueError("A evidência não corresponde ao trecho e às posições da redação confirmada.")

    def _validate_evaluation(self, raw, text, snapshot):
        evaluation = Evaluation.model_validate(raw)
        competencies = {item["id"]: item for item in snapshot["rubric"]["competencies"]}
        for item in evaluation.competencies:
            descriptor = next((d for d in competencies[item.id]["descriptors"]
                               if d["criteria_id"] == item.criteria_id and d["score"] == item.score), None)
            if descriptor is None:
                raise ValueError(f"O critério {item.criteria_id} não corresponde à competência e nota {item.id}/{item.score}.")
            expected = descriptor["citation"]
            if item.citation.page != expected["page"] or normalized(item.citation.excerpt) != normalized(expected["excerpt"]):
                raise ValueError("A citação da avaliação deve ser a citação do descritor oficial selecionado.")
            self.rubrics.validate_citation(item.citation, snapshot["pages"])
            for evidence in item.evidence:
                self._validate_evidence(evidence, text)
            for issue in item.issues:
                self._validate_evidence(issue.evidence, text)
        expected_facts = {rule["fact"] for rule in snapshot["rubric"]["special_rules"]}
        if {fact.id for fact in evaluation.facts} != expected_facts:
            raise ValueError("Avaliação incompleta: deve informar exatamente todos os fatos das regras especiais.")
        for fact in evaluation.facts:
            for evidence in fact.evidence:
                self._validate_evidence(evidence, text)
            if fact.value is True and not fact.evidence:
                raise ValueError("Fato positivo de regra especial exige evidência literal verificável.")
        return evaluation

    @staticmethod
    def _divergences(first, second):
        divergences = []
        fields = ("score", "criteria_id", "justification", "evidence", "issues")
        first_comp = {item.id: item.model_dump() for item in first.competencies}
        second_comp = {item.id: item.model_dump() for item in second.competencies}
        for identifier in sorted(COMPETENCY_IDS):
            changed = [field for field in fields if first_comp[identifier][field] != second_comp[identifier][field]]
            if changed:
                divergences.append({"id": identifier, "kind": "competency", "fields": changed,
                                    "first": first_comp[identifier], "second": second_comp[identifier]})
        second_facts = {item.id: item.model_dump() for item in second.facts}
        for fact in first.facts:
            if fact.model_dump() != second_facts[fact.id]:
                divergences.append({"id": f"fact:{fact.id}", "kind": "special_rule_fact",
                                    "first": fact.model_dump(), "second": second_facts[fact.id]})
        return divergences

    @staticmethod
    def _validate_context(rubric, theme, motivating_text, assessment_context):
        context = dict(assessment_context or {})
        context.update(theme=theme, motivating_text=motivating_text)
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("Informe o tema para verificar aderência e fuga ao tema.")
        for rule in rubric["special_rules"]:
            for requirement in rule.get("required_context", []):
                value = context.get(requirement)
                if value is None or value == "":
                    raise ValueError(f"Avaliação incompleta: falta informação '{requirement}' exigida pela regra '{rule['id']}'.")
                if requirement == "original_line_count" and (type(value) is not int or value < 0):
                    raise ValueError("A contagem de linhas originais precisa ser um inteiro confirmado.")
                if requirement == "original_image_review" and value is not True:
                    raise ValueError("Avaliação incompleta: confirme a revisão das imagens originais.")
        return context

    @staticmethod
    def _apply_rules(evaluation, snapshot):
        result = evaluation.model_dump()
        facts = {fact.id: fact for fact in evaluation.facts}
        applied = []
        for rule in snapshot["rubric"]["special_rules"]:
            fact = facts[rule["fact"]]
            if fact.value is None:
                raise ValueError(f"Avaliação incompleta: falta informação para decidir a regra '{rule['id']}'.")
            if not fact.value:
                continue
            applied.append({**deepcopy(rule), "fact_evidence": fact.model_dump()})
            for competency in result["competencies"]:
                if rule["action"] == "zero_total" or (rule["action"] == "zero_competency" and rule["competency_id"] == competency["id"]):
                    competency["score"] = 0
                elif rule["action"] == "cap_competency" and rule["competency_id"] == competency["id"]:
                    competency["score"] = min(competency["score"], rule["score"])
        for competency in result["competencies"]:
            original = next(item for item in evaluation.competencies if item.id == competency["id"])
            competency["normative_base_score"] = original.score
            competency["applied_rule_ids"] = [r["id"] for r in applied if r["action"] == "zero_total" or r["competency_id"] == competency["id"]]
            if original.score != competency["score"]:
                competency["justification"] = (f"Nota ajustada pela regra especial aprovada ({', '.join(competency['applied_rule_ids'])}). "
                                                f"Avaliação antes da regra: {original.justification}")
        result["scores"] = {item["id"]: item["score"] for item in result["competencies"]}
        result["total_score"] = sum(result["scores"].values())
        result["applied_rules"] = applied
        return result

    def correct(self, text, theme, motivating_text, prediction, assessment_context=None):
        audit = {"id": uuid.uuid4().hex, "started_at": datetime.now(timezone.utc).isoformat(),
                 "status": "running", "text_sha256": hashlib.sha256(str(text).encode()).hexdigest(),
                 "system_instructions": SYSTEM_INSTRUCTIONS, "calls": [], "model_prediction": deepcopy(prediction)}
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Confirme uma redação com texto antes da correção.")
            self._validate_prediction(prediction)
            snapshot = self.rubrics.get_rubric()
            audit["rubric_snapshot"] = snapshot
            context = self._validate_context(snapshot["rubric"], theme, motivating_text, assessment_context)
            retrieved = self.rubrics.retrieve(f"Tema: {theme}\nRedação: {text}", snapshot=snapshot)
            audit["retrieved_sources"] = retrieved
            common = {"task": "evaluate_essay", "rubric": snapshot["rubric"],
                      "rubric_version": snapshot["version"], "retrieved_sources": retrieved,
                      "auxiliary_supervised_prediction": prediction,
                      "untrusted_essay_data": {"text": text, "theme": theme, "motivating_text": motivating_text,
                                               "confirmed_assessment_context": context},
                      "required_fact_ids": sorted({rule["fact"] for rule in snapshot["rubric"]["special_rules"]})}
            evaluations = []
            for round_number, seed in [(1, 11), (2, 29)]:
                payload = {**deepcopy(common), "independent_round": round_number,
                           "instruction": "Leia toda a redação e aplique cada descritor e regra. Faça uma avaliação independente, sem presumir notas anteriores. Gere justificativas específicas, evidências exatas e propostas de melhoria por competência."}
                call = {"phase": f"independent_{round_number}", "seed": seed, "model": self.llm.model, "request": payload}
                audit["calls"].append(call)
                raw = self.llm.generate(payload, Evaluation.model_json_schema(), seed=seed)
                call["raw_response"] = deepcopy(raw)
                evaluations.append(self._validate_evaluation(raw, text, snapshot))
            first, second = evaluations
            divergences = self._divergences(first, second)
            resolved = first
            resolution = None
            if divergences:
                payload = {**deepcopy(common), "task": "review_divergences", "divergences": divergences,
                           "first_evaluation": first.model_dump(), "second_evaluation": second.model_dump(),
                           "instruction": "Releia a redação e a rubrica. Resolva cada divergência de nota, justificativa ou fato com base na fonte normativa. Não calcule médias. Retorne avaliação completa validável e uma resolução fundamentada para cada ID divergente; não altere notas sem divergência."}
                call = {"phase": "review", "seed": 47, "model": self.llm.model, "request": payload}
                audit["calls"].append(call)
                raw = self.llm.generate(payload, Review.model_json_schema(), seed=47)
                call["raw_response"] = deepcopy(raw)
                review = Review.model_validate(raw)
                required = {item["id"] for item in divergences}
                if {item.id for item in review.resolutions} != required or len(review.resolutions) != len(required):
                    raise ValueError("A revisão deve justificar exatamente todas as divergências.")
                resolved = self._validate_evaluation(review.evaluation.model_dump(), text, snapshot)
                for item in resolved.competencies:
                    previous = next(c for c in first.competencies if c.id == item.id)
                    if item.id not in required and item.score != previous.score:
                        raise ValueError("A revisão alterou uma nota sem divergência nem resolução fundamentada.")
                resolution = review.model_dump()
                evaluations.append(resolved)
            result = self._apply_rules(resolved, snapshot)
            audit.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
            result.update({"rubric_info": {"year": 2026, "version": snapshot["version"],
                                           "document_sha256": snapshot["document_sha256"], "reviewer": snapshot["reviewer"],
                                           "approved_at": snapshot["approved_at"]},
                           "evaluations": [item.model_dump() for item in evaluations], "divergences": divergences,
                           "review": resolution, "model_prediction": deepcopy(prediction), "audit": audit,
                           "disclaimer": DISCLAIMER, "consistency": {"status": "experimental",
                               "independent_passes": 2, "divergence_count": len(divergences),
                               "reviewed": bool(divergences), "same_local_model": True}})
            return result
        except (ValueError, RuntimeError) as exc:
            audit.update(status="failed", error=str(exc), completed_at=datetime.now(timezone.utc).isoformat())
            if isinstance(exc, RuntimeError):
                raise CorrectionUnavailable(str(exc), audit) from exc
            raise CorrectionError(str(exc), audit) from exc
