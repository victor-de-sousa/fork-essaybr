"""Normative fixtures are synthetic and used only by this test suite."""
from types import SimpleNamespace

import numpy as np
import pymupdf as fitz
import pytest

from app.services.rubric_service import RubricService


def settings_at(tmp_path):
    return SimpleNamespace(root_dir=tmp_path, rubric_dir=tmp_path / "rubrics",
                           embedding_model_dir=tmp_path / "no-embedding-model",
                           ollama_url="http://127.0.0.1:11434", ollama_model="test-only")


def source_pdf(year=2026):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 40), f"ENEM {year} - Cartilha do participante - TEST FIXTURE")
    for index in range(5):
        page.insert_text((40, 80 + index * 80), f"Competencia {index + 1}")
        for level, score in enumerate([0, 40, 80, 120, 160, 200]):
            page.insert_text((40, 94 + index * 80 + level * 10), f"C{index + 1} nivel {score}: descritor normativo {level}", fontsize=8)
    page = doc.new_page()
    page.insert_text((40, 40), "Fuga ao tema implica nota zero em todas as competencias.")
    page.insert_text((40, 60), "Exemplo oficial separado da matriz.")
    data = doc.tobytes()
    doc.close()
    return data


def reviewed_payload(ingestion):
    return {
        "document_sha256": ingestion["document_sha256"],
        "official_source_confirmed": True, "special_rules_reviewed": True,
        "reviewer": "Revisor de teste",
        "rubric": {
            "year": 2026, "title": "ENEM 2026", "identity_citation": {
                "page": 1, "excerpt": "ENEM 2026 - Cartilha do participante"},
            "competencies": [{
                "id": f"c{i}", "name": f"Competencia {i}",
                "descriptors": [{"score": score, "criteria_id": f"c{i}_{score}",
                                 "description": f"Descritor {score}",
                                 "citation": {"page": 1, "excerpt": f"C{i} nivel {score}: descritor normativo {j}"}}
                                for j, score in enumerate([0, 40, 80, 120, 160, 200])]
            } for i in range(1, 6)],
            "special_rules": [{"id": "off_topic", "description": "Fuga ao tema",
                               "fact": "off_topic", "action": "zero_total", "competency_id": None,
                               "score": None, "citation": {"page": 2, "excerpt": "Fuga ao tema implica nota zero em todas as competencias."}}],
            "examples": [{"text": "Exemplo oficial separado da matriz.", "citation": {
                "page": 2, "excerpt": "Exemplo oficial separado da matriz."}}],
        },
    }


@pytest.fixture
def ingested(tmp_path, monkeypatch):
    settings = settings_at(tmp_path)
    service = RubricService(settings)
    def unavailable(*args, **kwargs):
        raise RuntimeError("Ollama indisponível no teste")
    monkeypatch.setattr(service.llm, "generate", unavailable)
    ingestion = service.ingest(source_pdf(), "cartilha.pdf")
    return settings, service, ingestion


def test_extracts_actual_pages_without_ollama_and_requires_review(ingested):
    _, service, result = ingested
    assert result["page_count"] == 2
    assert result["pages"][1]["page"] == 2
    assert "Fuga ao tema" in result["pages"][1]["text"]
    assert result["status"] == "pending_review"
    assert result["proposal"] is None
    assert "indisponível" in result["extraction_error"]
    with pytest.raises(ValueError, match="aprovada"):
        service.get_rubric()


def test_verified_rubric_is_immutable_when_new_pdf_is_ingested(ingested):
    _, service, result = ingested
    service.approve(reviewed_payload(result))
    snapshot = service.get_rubric()
    assert snapshot["rubric"]["competencies"][0]["id"] == "c1"
    assert snapshot["document_sha256"] == result["document_sha256"]
    service.ingest(source_pdf(), "nova-cartilha.pdf")
    assert service.get_version(snapshot["version"]) == snapshot
    with pytest.raises(ValueError, match="aprovada"):
        service.get_rubric()


@pytest.mark.parametrize("tamper", ["citation", "page", "year", "score", "duplicate", "missing_rule", "confirmation", "hash"])
def test_rejects_unverified_or_incomplete_normative_payload(ingested, tamper):
    _, service, result = ingested
    payload = reviewed_payload(result)
    rubric = payload["rubric"]
    if tamper == "citation": rubric["competencies"][0]["descriptors"][0]["citation"]["excerpt"] = "Texto inventado"
    if tamper == "page": rubric["competencies"][0]["descriptors"][0]["citation"]["page"] = 99
    if tamper == "year": rubric["year"] = 2025
    if tamper == "score": rubric["competencies"][0]["descriptors"][0]["score"] = 20
    if tamper == "duplicate": rubric["competencies"][1]["id"] = "c1"
    if tamper == "missing_rule": rubric["special_rules"] = []
    if tamper == "confirmation": payload["official_source_confirmed"] = False
    if tamper == "hash": payload["document_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        service.approve(payload)
    assert service.status()["status"] == "pending_review"


def test_filename_does_not_override_wrong_pdf_year(tmp_path, monkeypatch):
    service = RubricService(settings_at(tmp_path))
    monkeypatch.setattr(service.llm, "generate", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    result = service.ingest(source_pdf(2025), "cartilha_oficial_2026.pdf")
    with pytest.raises(ValueError, match="citação|2026"):
        service.approve(reviewed_payload(result))


def test_missing_local_embedding_never_falls_back_to_fake_similarity(ingested):
    _, service, result = ingested
    service.approve(reviewed_payload(result))
    with pytest.raises(RuntimeError, match="embedding"):
        service.retrieve("competencia escrita")


def test_rejects_json_export_as_pdf(tmp_path):
    service = RubricService(settings_at(tmp_path))
    with pytest.raises(ValueError, match="PDF"):
        service.ingest(b'{"pages": []}', "cartilha_enem_2026.json")


def test_ingests_structured_2026_json_by_page(tmp_path, monkeypatch):
    service = RubricService(settings_at(tmp_path))
    monkeypatch.setattr(service.llm, "generate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    payload = b'{"schemaVersion": 5, "pages": [{"children": [{"text": "ENEM 2026"}, {"children": [{"text": "Competencia 1"}]}]}]}'
    result = service.ingest(payload, "cartilha_enem_2026.json")
    assert result["status"] == "pending_review"
    assert result["source_format"] == "json"
    assert result["page_count"] == 1
    assert "Competencia 1" in result["pages"][0]["text"]


def test_tampered_source_revokes_approval(ingested):
    _, service, result = ingested
    service.approve(reviewed_payload(result))
    source = service.directory / "sources" / f"{result['document_sha256']}.pdf"
    source.write_bytes(source_pdf(2025))
    with pytest.raises(ValueError, match="alterado"):
        service.get_rubric()


def test_semantic_retrieval_uses_all_tokens_and_returns_original_page(ingested, monkeypatch):
    settings, service, result = ingested
    service.approve(reviewed_payload(result))
    settings.embedding_model_dir.mkdir()
    (settings.embedding_model_dir / "test-weights").write_text("test fixture")

    class TestEmbedding:
        max_seq_length = 64

        def tokenizer(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

        def encode(self, texts, **kwargs):
            if any(len(text) > self.max_seq_length for text in texts):
                raise AssertionError("Would silently truncate in the real encoder")
            return np.asarray([[1.0, 0.0] if "Fuga" in text else [0.0, 1.0] for text in texts])

    monkeypatch.setattr(service, "_embedding_model", lambda: TestEmbedding())
    hits = service.retrieve("Fuga ao tema", limit=1)
    assert hits[0]["page"] == 2
    assert "Fuga" in hits[0]["text"]
    assert hits[0]["text"] in result["pages"][1]["text"]
    assert service.retrieve("x" * 200 + "Fuga", limit=2)
