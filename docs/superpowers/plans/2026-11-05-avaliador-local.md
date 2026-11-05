# Avaliador ENEM local — plano de implementação

> Execução autônoma solicitada explicitamente no anexo. Implementação em paralelo por subsistemas, com testes e revisão final. O workspace atual é o destino solicitado; preservar os arquivos do corpus e o JSON de cartilha existente. Sem commits, publicação ou alteração do corpus original.

**Objetivo:** construir um pipeline treinável e uma aplicação local com transcrição confirmada, previsão supervisionada e avaliações justificadas pela cartilha carregada.

**Arquitetura:** FastAPI + SQLAlchemy/SQLite; Vue 3/Vite; pipeline ML independente; OCR TrOCR local segmentado por linhas; Ollama apenas em loopback; PyMuPDF e embeddings locais com rubrica explícita, citações verificadas e revisão humana da extração normativa. Falhar claramente na ausência de modelos e cartilha.

**Restrições:** Python >=3.11, sem APIs externas de IA, sem notas simuladas, escala 0..200/40, sem truncamento silencioso, teste reservado, consentimento para reutilizar referências humanas. Downloads são configuração explícita; inferência não baixa modelos.

## Inspeção realizada

- Apenas corpus Essay-BR e scripts originais; sem frontend ou backend.
- CSV: prompt, title, essay (lista de parágrafos), competence (cinco notas), score. Prompts separados. Splits training/development/testing existentes.
- Leitor legado usa eval: substituir por ast.literal_eval mantendo interface.
- cartilha_enem_2026.json é uma composição gráfica de 82 páginas, não um PDF oficial autenticado. Preservar; não autorizar pontuação a partir do nome do arquivo.
- Ambiente Python 3.12; dependências ML/API ausentes no sandbox; disponibilidade de Node/Ollama/GPU será verificada também no host quando necessário.

## Tarefas e provas

- [ ] Dados/ML: carregar splits reais, auditoria de dados inválidos e duplicação aproximada; TF-IDF/Ridge real; Transformer por segmentos, ordinal/regressão; treino/checkpoints/early stopping; métricas/seleção somente na validação; avaliação final explícita. Testes de dados, métricas, treino e recarga.
- [ ] Normas: extração PDF, versão/hash/páginas, proposta de rubrica estruturada via Ollama, aprovação humana de descritores e exceções, índice semântico local. Duas avaliações independentes, revisão de divergências, validação de evidências e referências, auditoria preservada. Testes com dependências substituídas apenas na suíte.
- [ ] API/OCR: imagens validadas e ordenadas, persistência relacional, OCR local com incerteza, confirmação explícita, erros consistentes, histórico/exclusão, comparação humana consentida. Testes unitários e integração completa.
- [ ] Frontend: início, envio/reordenação, revisão lado a lado, relatório/evidências, histórico, configuração da cartilha, notas humanas. Build e testes de comportamento com respostas de API de teste.
- [ ] Operação: instalação, download local, diagnóstico, preparo/treino/comparação/avaliação, execução, README e relatório real de verificação.

## Contratos entre subsistemas

- Backend importável como `app` com PYTHONPATH=backend.
- ML: `app.ml.predict.Predictor(model_dir: Path)`; `.predict(text: str) -> dict` com `scores: {c1..c5}`, `model_info: {version, trained, kind}`, `.status() -> dict`. Modelo ausente gera exceção clara. `app.ml.metrics.score_metrics(y_true,y_pred)` reutilizado nas comparações humanas.
- Correção: `app.services.correction_service.CorrectionService(settings)`; `.correct(text, theme, motivating_text, prediction) -> dict` com scores, total_score, competencies, general_feedback, rubric_info, evaluations, divergences, disclaimer. Configuração via settings.root_dir, settings.rubric_dir, settings.embedding_model_dir, settings.ollama_url, settings.ollama_model.
- Rubrica: `app.services.rubric_service.RubricService(settings)`; `.ingest(pdf_bytes, filename) -> dict`, `.status() -> dict`, `.approve(payload: dict) -> dict`, `.get_rubric() -> dict`. Endpoints POST /api/rubric/upload (file), GET /api/rubric, PUT /api/rubric/approve (rubrica editada); ingestão retorna proposta ou páginas/estado de extração.
- Essays JSON: id, theme, motivating_text, status, created_at, transcription, confirmed, ocr_warnings, images [{id, position, url}], correction (dict|null), error (string|null).
- POST /api/essays/upload multipart files[] via campo `files`, theme, motivating_text; POST /{id}/transcribe; PATCH /{id}/transcription {text,confirmed,theme?,motivating_text?}; POST /{id}/correct; GET /{id}; GET /{id}/correction; GET /api/essays retorna lista; DELETE /{id}.
- GET /api/model/status: scoring, ocr, ollama, rubric. POST /api/essays/{id}/reference {scores:{c1..c5}, notes, consent_training:false}; GET /api/validation/report.
- Erros: {error:{code,message}}. Inferências bloqueantes executadas em handlers síncronos FastAPI. URLs de imagens controladas /api/essays/{id}/images/{image_id}.

## Foco de revisão

1. Fonte ausente/ilegível/ano inconsistente/citação que não existe: bloquear correção.
2. Modificação da transcrição após avaliação: invalidar relatório corrente, manter auditoria.
3. Vazamento de textos duplicados ou referências humanas de teste para treinamento: detectar e impedir.
4. Imagens malformadas ou nomes maliciosos: não escrever caminhos arbitrários nem deixar registros parciais.
5. LLM retorna JSON inválido, notas inválidas ou evidências inventadas: não persistir como correção concluída.

## Registro de execução

Atualizar este documento e docs/verification.md com provas e limitações concretas.
