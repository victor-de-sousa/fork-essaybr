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
- A inferência do baseline foi executada localmente com o artefato versionado,
  e a avaliação do teste reservado foi reexecutada sem alterar o modelo.
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
- Foi baixado o checkpoint oficial `neuralmind/bert-base-portuguese-cased` e
  executado um fine-tuning ordinal de engenharia em CPU. O checkpoint real está
  em `artifacts/models/bertimbau-ordinal-smoke`, com pesos safetensors,
  tokenizer, log e estado do otimizador.
- O smoke training usou somente 16 redações de treino, 8 de desenvolvimento,
  uma época, janela de 64 tokens e sobreposição de 8. Seu metadata contém
  `trained: false` e `engineering_smoke: true`; portanto o Predictor bloqueia
  esse artefato e ele não é um modelo de produção.
- Teste reservado do baseline, reexecutado nesta etapa: MAE médio 21,1312,
  RMSE médio 34,1778 e QWK médio 0,4470. O relatório está em
  `artifacts/reports/baseline_final_test_recheck.json`; consulte também
  `docs/ml.md` e os relatórios para métricas por competência.
- BERTimbau, OCR e embeddings não têm artefato treinado/configurado neste
- checkout. Há somente o checkpoint BERTimbau de engenharia descrito acima.

## Próximas tarefas

1. Revisar a rubrica da cartilha 2026 no fluxo de Configurações e aprová-la com
   citações literais verificáveis.
2. Instalar/configurar modelos locais de OCR e embeddings, sem downloads
   implícitos na aplicação.
3. Executar avaliação independente após qualquer novo treinamento.
4. Adicionar um conjunto de amostras manuscritas anotadas para validar OCR e
   revisão da transcrição.
5. Obter explicitamente um checkpoint local BERTimbau compatível e executar um
   treino limitado com tempo/memória configuráveis; comparar somente após
   registrar a seleção no conjunto de desenvolvimento.

## Validação executada em 2026-11-05

- Backend: 69 testes Python aprovados.
- Frontend: 15 testes Vitest aprovados.
- Frontend: build de produção Vite aprovado.
- Inferência real do baseline: artefato `tfidf-ridge-5d3e145e9996`,
  previsão de exemplo `[120, 120, 80, 120, 80]`.
- Fine-tuning Transformer de produção: **não concluído**. A tentativa com
  3.194 redações excedeu o limite de 30 minutos em CPU antes de concluir uma
  época e não gerou checkpoint parcial.
- Fine-tuning Transformer de engenharia: concluído com BERTimbau ordinal,
  checkpoint em `artifacts/models/bertimbau-ordinal-smoke`. A inferência direta
  funcionou e produziu `[80, 80, 80, 80, 80]`; o Predictor rejeita o artefato
  por `trained: false`.
- Métricas do smoke validation (8 exemplos): MAE médio 47,0, RMSE médio
  54,6680 e QWK médio 0,0. Esses números não representam qualidade do modelo
  e não foram usados para seleção.
- Avaliação final reexecutada no conjunto `testing` com a flag explícita
  `--final-test`; nenhum hiperparâmetro foi ajustado.
- O build reporta um aviso de CSS sobre `@import none`, sem impedir a geração
  dos arquivos.
- `npm install` reportou vulnerabilidades nas dependências transitivas. Não foi
  executado `npm audit fix` automaticamente para evitar atualizações
  incompatíveis.

## Decisões técnicas

- A cartilha aprovada é imutável e versionada por hash do documento e do
  conteúdo estruturado.
- A fonte normativa é prioritária sobre a previsão estatística; o modelo
  supervisionado serve apenas como evidência auxiliar.
- Notas são aceitas somente na escala `0, 40, 80, 120, 160, 200`, e toda
  avaliação exige evidências literais, citações da rubrica e soma consistente.
