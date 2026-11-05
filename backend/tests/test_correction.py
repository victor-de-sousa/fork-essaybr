import httpx
import json
import pytest

from app.schemas.correction import Evaluation
from app.services.correction_service import CorrectionService
from app.services.feedback_service import OllamaClient
from test_rubric import ingested, reviewed_payload, settings_at


TEXT = "A educação pública precisa de investimento. Assim, o governo deve financiar bibliotecas."


def evaluation():
    return {
        "competencies": [{"id": f"c{i}", "score": 120, "criteria_id": f"c{i}_120",
            "justification": "O argumento permite avaliar este descritor.",
            "evidence": [{"text": "educação pública", "start": 2, "end": 18}],
            "issues": [{"kind": "argumentacao", "explanation": "Detalhar a proposta.",
                        "evidence": {"text": "bibliotecas", "start": 76, "end": 87}, "suggestion": "Explicar os meios."}],
            "strengths": ["Apresenta proposta."], "improvements": ["Detalhar os meios."],
            "citation": {"page": 1, "excerpt": f"C{i} nivel 120: descritor normativo 3"}}
            for i in range(1, 6)],
        "total_score": 600, "general_feedback": "Desenvolva argumentos e detalhes da intervenção.",
        "facts": [{"id": "off_topic", "value": False, "explanation": "O texto trata do tema educação.",
                   "evidence": [{"text": "educação pública", "start": 2, "end": 18}]}],
    }


@pytest.fixture
def correction_ready(ingested, monkeypatch):
    settings, rubric, result = ingested
    rubric.approve(reviewed_payload(result))
    service = CorrectionService(settings)
    monkeypatch.setattr(service.rubrics, "retrieve", lambda query, **kw: [{"page": 1, "text": "ENEM 2026", "similarity": .8}])
    return service


def prediction():
    return {"scores": {f"c{i}": 160 for i in range(1, 6)}, "model_info": {"version": "trained-fixture", "trained": True, "kind": "test"}}


@pytest.mark.parametrize("change", ["score", "total", "duplicate", "extra"])
def test_evaluation_rejects_invalid_scores_totals_and_ids(change):
    candidate = evaluation()
    if change == "score": candidate["competencies"][0]["score"] = 199
    if change == "total": candidate["total_score"] = 1000
    if change == "duplicate": candidate["competencies"][1]["id"] = "c1"
    if change == "extra": candidate["unexpected"] = "ignore"
    with pytest.raises(ValueError):
        Evaluation.model_validate(candidate)


def test_requires_real_trained_prediction_before_generating_feedback(correction_ready):
    with pytest.raises(ValueError, match="treinado"):
        correction_ready.correct(TEXT, "Educação", "", {"scores": {}, "model_info": {"trained": False}})


def test_two_independent_passes_preserve_normative_scores_and_audit(correction_ready, monkeypatch):
    calls = []
    def generate(payload, schema, **kwargs):
        calls.append(payload)
        return evaluation()
    monkeypatch.setattr(correction_ready.llm, "generate", generate)
    result = correction_ready.correct(TEXT, "Educação", "", prediction())
    assert len(calls) == 2
    assert "previous_evaluation" not in calls[1]
    assert result["total_score"] == 600
    assert result["scores"]["c1"] == 120
    assert len(result["evaluations"]) == 2
    assert result["model_prediction"]["scores"]["c1"] == 160
    assert result["rubric_info"]["document_sha256"]
    assert "experimental" in result["disclaimer"].lower()


@pytest.mark.parametrize("change", ["quote", "position", "citation", "criteria"])
def test_fabricated_evidence_or_citation_never_becomes_final_grade(correction_ready, monkeypatch, change):
    answer = evaluation()
    if change == "quote": answer["competencies"][0]["evidence"][0]["text"] = "frase inexistente"
    if change == "position": answer["competencies"][0]["evidence"][0]["start"] = 3
    if change == "citation": answer["competencies"][0]["citation"]["excerpt"] = "regra inventada"
    if change == "criteria": answer["competencies"][0]["criteria_id"] = "c2_120"
    monkeypatch.setattr(correction_ready.llm, "generate", lambda *a, **kw: answer)
    with pytest.raises(ValueError, match="evidência|citação|critério"):
        correction_ready.correct(TEXT, "Educação", "", prediction())


def test_disagreement_is_reviewed_against_rubric_not_averaged(correction_ready, monkeypatch):
    first, second = evaluation(), evaluation()
    second["competencies"][0].update(score=160, criteria_id="c1_160", justification="Há desenvolvimento mais completo.", citation={"page": 1, "excerpt": "C1 nivel 160: descritor normativo 4"})
    second["total_score"] = 640
    responses = iter([first, second, {"evaluation": second, "resolutions": [{"id": "c1", "reason": "O descritor de 160 é mais aderente aos argumentos do texto."}]}])
    monkeypatch.setattr(correction_ready.llm, "generate", lambda *a, **kw: next(responses))
    result = correction_ready.correct(TEXT, "Educação", "", prediction())
    assert result["total_score"] == 640
    assert result["scores"]["c1"] == 160
    assert result["divergences"][0]["id"] == "c1"
    assert len(result["evaluations"]) == 3


def test_special_zero_rule_overrides_all_estimates(correction_ready, monkeypatch):
    answer = evaluation()
    answer["facts"][0]["value"] = True
    monkeypatch.setattr(correction_ready.llm, "generate", lambda *a, **kw: answer)
    result = correction_ready.correct(TEXT, "Saúde", "", prediction())
    assert result["total_score"] == 0
    assert set(result["scores"].values()) == {0}
    assert result["applied_rules"][0]["id"] == "off_topic"


def test_unknown_special_rule_fact_fails_closed(correction_ready, monkeypatch):
    answer = evaluation()
    answer["facts"][0]["value"] = None
    monkeypatch.setattr(correction_ready.llm, "generate", lambda *a, **kw: answer)
    with pytest.raises(ValueError, match="incompleta|informação"):
        correction_ready.correct(TEXT, "Educação", "", prediction())


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://192.168.1.10:11434", "http://localhost.evil.example", "http://user:pass@127.0.0.1:11434", "http://127.0.0.1:11434/api?q=x"])
def test_llm_client_rejects_nonlocal_or_ambiguous_urls(tmp_path, url):
    settings = settings_at(tmp_path)
    settings.ollama_url = url
    with pytest.raises(ValueError, match="loopback|local"):
        OllamaClient(settings)


def test_local_transport_never_follows_remote_redirect(tmp_path, monkeypatch):
    real_client = httpx.Client
    def respond(request):
        if request.url.host != "127.0.0.1":
            pytest.fail("Essay data redirected to an external host")
        return httpx.Response(307, headers={"Location": "https://example.org/collect"})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(RuntimeError, match="Ollama"):
        OllamaClient(settings_at(tmp_path)).generate({"essay": TEXT}, {})


@pytest.mark.parametrize("response", ["not json", '{"message":{"content":"{bad}"}}', '{"message":{"content":"{\\"score\\":40,\\"score\\":200}"}}'])
def test_malformed_or_ambiguous_ollama_json_is_rejected(tmp_path, monkeypatch, response):
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=response))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    with pytest.raises(ValueError, match="JSON"):
        OllamaClient(settings_at(tmp_path)).generate({"essay": TEXT}, {})


def test_unresolved_review_never_publishes_partial_grade(correction_ready, monkeypatch):
    first, second = evaluation(), evaluation()
    second["competencies"][0]["justification"] = "Justificativa divergente exige revisão."
    responses = iter([first, second, {"evaluation": second, "resolutions": [{"id": "c2", "reason": "Omitiu o conflito real."}]}])
    monkeypatch.setattr(correction_ready.llm, "generate", lambda *a, **kw: next(responses))
    with pytest.raises(ValueError, match="divergências") as captured:
        correction_ready.correct(TEXT, "Educação", "", prediction())
    assert captured.value.audit["status"] == "failed"
    assert len(captured.value.audit["calls"]) == 3
    assert captured.value.audit["calls"][2]["raw_response"]["resolutions"][0]["id"] == "c2"


def test_missing_physical_line_context_cannot_be_inferred_from_paragraphs(ingested, monkeypatch):
    settings, rubric, result = ingested
    payload = reviewed_payload(result)
    payload["rubric"]["special_rules"][0]["required_context"] = ["original_line_count"]
    rubric.approve(payload)
    service = CorrectionService(settings)
    with pytest.raises(ValueError, match="original_line_count"):
        service.correct(TEXT, "Educação", "", prediction())
