# Estado do projeto

## Estado atual

- Backend FastAPI, persistência SQLite, upload de imagens, histórico e edição de
  transcrição estão implementados.
- OCR, predição supervisionada e correção fundamentada possuem contratos
  explícitos e bloqueiam resultados quando os modelos ou a cartilha aprovada
  não estão disponíveis.
- O baseline TF-IDF/Ridge foi treinado com o Essay-BR preparado, com teste
  reservado e métricas registradas em `artifacts/reports/`.
- A interface Vue 3 cobre upload, revisão, resultado, histórico e configuração
  local.
- A cartilha pode ser ingerida a partir do PDF original ou do JSON estruturado
  existente em `cartilha_enem_2026.json`. O JSON é convertido em páginas de
  texto mantendo a ordem dos nós e é identificado por SHA-256.

## Funcionalidades incompletas

- A rubrica ainda exige revisão e aprovação humana; a ingestão não transforma a
  publicação em critérios aprovados automaticamente.
- Não há checkpoint Transformer treinado neste workspace.
- OCR de escrita manuscrita e embeddings dependem de modelos locais instalados
  pelo operador.

## Problemas conhecidos e limitações

- O JSON estruturado contém o texto da publicação, não uma rubrica já validada.
  Citações, descritores e regras precisam ser conferidos antes da aprovação.
- O LLM local é opcional para propor a estrutura; indisponibilidade deixa a
  fonte em `pending_review`, sem produzir nota.
- As métricas disponíveis são do baseline e não comprovam equivalência à
  correção oficial do INEP.

## Modelos e métricas

- Baseline: TF-IDF unigramas/bigramas + Ridge multissaída, artefato em
  `artifacts/models/baseline`.
- Teste reservado do baseline: MAE médio 21,1312, RMSE médio 34,1778 e QWK
  médio 0,4470. Consulte `docs/ml.md` e os relatórios para métricas por
  competência.
- BERTimbau, OCR e embeddings não têm artefato treinado/configurado neste
  checkout.

## Próximas tarefas

1. Revisar a rubrica da cartilha 2026 no fluxo de Configurações e aprová-la com
   citações literais verificáveis.
2. Instalar/configurar modelos locais de OCR e embeddings, sem downloads
   implícitos na aplicação.
3. Executar avaliação independente após qualquer novo treinamento.
4. Adicionar um conjunto de amostras manuscritas anotadas para validar OCR e
   revisão da transcrição.

## Decisões técnicas

- A cartilha aprovada é imutável e versionada por hash do documento e do
  conteúdo estruturado.
- A fonte normativa é prioritária sobre a previsão estatística; o modelo
  supervisionado serve apenas como evidência auxiliar.
- Notas são aceitas somente na escala `0, 40, 80, 120, 160, 200`, e toda
  avaliação exige evidências literais, citações da rubrica e soma consistente.
