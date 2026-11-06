# Auditoria de QWK e CPU

Escopo autorizado: auditoria, correções comprovadas, testes e benchmarks curtos.
Não executar treino completo; preservar modelos, backups, logs e corpus existentes.
Não selecionar nada com o teste independente.

1. Inspecionar os sete módulos ML, metadados, confusões e estados de retomada.
2. Recalcular QWK por fórmula independente e comparar baseline na validação.
3. Medir CPU real: tokenização, seleção, padding, encoder, cabeça e validação,
   com amostra aleatória reproduzível e diferentes quantidades de threads.
4. Testar antes de corrigir: escala/ordinal, preservação de ordem e máscaras,
   equivalência de logits/gradientes online e cache, invalidação por pesos/textos,
   dropout congelado, retomada ao fim e no meio da época, proteção do output.
5. Extrair métodos de features/cabeça; cache persistente FP32 por conteúdo dos
   pesos, configuração, tokenizer, textos e pooling. Permitir apenas frozen.
   Integrar opt-in ao treino, armazenar também validação, registrar tempos.
6. Corrigir estado de retomada e impedir substituição acidental de experimento.
   Manter LR/decodificação atuais; registrar alternativas como experimentos.
7. Rodar benchmark pareado online/cache, verificação do backend e relatório
   com evidências, limites da extrapolação e plano de experimentos.

Riscos a verificar: padding variável; alteração do encoder sob mesmo caminho;
retomada sem melhor modelo; compatibilidade dos checkpoints antigos; cache
indevido em full; disponibilidade do modelo/tokenizer na inferência.
