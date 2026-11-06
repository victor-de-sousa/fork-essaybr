# Pipeline supervisionado local

Os CSV originais do [Essay-BR](../essay-br/) são preservados. O programa não inventa notas se não houver artefato treinado. O baseline disponível foi ajustado realmente ao corpus; o Transformer exige download local explícito e treinamento. Os números do baseline não representam desempenho de BERTimbau.

## Dados e prevenção de vazamento

```bash
PYTHONPATH=backend .venv/bin/python -m app.ml.dataset \
  --source essay-br/splits --output datasets/prepared
```

O [preparador](../backend/app/ml/dataset.py) usa `ast.literal_eval`, mantém os parágrafos separados por duas quebras de linha e rejeita notas que não sejam exatamente cinco inteiros em `{0,40,80,120,160,200}` ou cujo total não corresponda à soma. Título e prompt permanecem como metadados; o baseline e o Transformer usam o corpo da redação como entrada.

A auditoria registra hashes SHA256 dos CSV e JSONL, contagens, distribuição de notas/temas, comprimentos, linhas inválidas e grupos de duplicação. A aproximação usa Jaccard de trigramas de palavras com limiar fixo de 0,85, normalização Unicode e caixa; uma filtragem por prefixo global gera candidatos e a comparação final calcula Jaccard exato. É uma política conservadora de similaridade textual, sem garantia de detectar paráfrases semânticas.

Um grupo que atinja teste fica reservado ao teste; seus membros em treino/validação vão para quarentena. Sem teste, validação tem prioridade sobre treino. Duplicatas internas de treino/validação são reduzidas a um representante. Duplicatas internas de teste são relatadas e preservadas para não alterar a população reservada. Os rótulos de teste nunca selecionam hiperparâmetros. Apenas os textos são comparados para impedir vazamento.

Execução neste workspace: 4.570 redações; 0 inválidas; 4 redações de treino em quarentena, sendo três grupos próximos entre splits e uma duplicata exata interna ao treino. Resultado: **3.194 treino / 686 validação / 686 teste**. Veja [auditoria](../datasets/prepared/audit.json), [manifesto](../datasets/prepared/manifest.json) e [quarentena](../datasets/prepared/quarantine.jsonl). Antes do treinamento, os hashes de treino e validação são conferidos.

## Baseline reproduzível

```bash
PYTHONPATH=backend .venv/bin/python -m app.ml.baseline \
  --data datasets/prepared --output artifacts/models/baseline \
  --alphas 1 10 100 --seed 42
```

TF-IDF de unigramas/bigramas, até 50.000 atributos e TF sublinear, alimenta Ridge multissaída. Vocabulário e pesos são ajustados apenas ao treino. O maior QWK médio na validação seleciona alpha; MAE médio desempata. O artefato contém pipeline completo, versões, hashes, parâmetros e métricas. `joblib` deve carregar somente artefatos criados localmente e confiáveis; sua desserialização pode executar código. A inferência confere o hash registrado.

## BERTimbau: ordinal e regressão

Após baixar `neuralmind/bert-base-portuguese-cased` para a pasta configurada, execute:

```bash
PYTHONPATH=backend .venv/bin/python -m app.ml.train \
  --base-model artifacts/models/bertimbau --data datasets/prepared \
  --output artifacts/models/bertimbau-ordinal --objective ordinal \
  --epochs 8 --batch-size 2 --learning-rate 2e-5 --patience 2 --seed 42

PYTHONPATH=backend .venv/bin/python -m app.ml.train \
  --base-model artifacts/models/bertimbau --data datasets/prepared \
  --output artifacts/models/bertimbau-regression --objective regression \
  --epochs 8 --batch-size 2 --learning-rate 2e-5 --patience 2 --seed 42
```

Todo o encoder é ajustado por AdamW com warmup linear de 10%, decaimento linear, regularização de 0,01 e clipping de gradiente em 1,0. Há cinco saídas por redação. Na versão ordinal, cada competência possui cinco limiares cumulativos monotônicos, otimizados por BCE; a decodificação retorna a mediana ordinal. Na regressão, sigmoide limita a faixa e MSE usa alvos normalizados para 0..1. Não há regras textuais simulando aprendizado.

Textos são tokenizados com `truncation=False`, divididos em segmentos de até 512 tokens incluindo especiais, com sobreposição de 64. Todos os segmentos, inclusive o último incompleto, são codificados. As representações CLS são agregadas por média ponderada pelo comprimento para produzir uma nota por competência da redação inteira. O tamanho dos minibatches do encoder é controlado por `--chunk-batch-size`; textos muito longos podem exigir memória adicional durante retropropagação. A sobreposição dá peso adicional às regiões sobrepostas; não substitui contexto global de longa distância.

Seeds de Python/NumPy/PyTorch são fixados; operações determinísticas são solicitadas, com aviso para operações sem implementação determinística. Diferentes versões de bibliotecas/hardware ainda podem alterar resultados. Há checkpoints em `safetensors`, tokenizador local, logs JSON por época e estado de otimizador/scheduler/RNG em `checkpoint-last/trainer.pt`. `--resume` exige os mesmos dados e hiperparâmetros. O melhor checkpoint e early stopping usam apenas validação. A inferência usa `local_files_only=True`; nunca baixa pesos implicitamente.

O teste de engenharia usa BERT minúsculo inicializado aleatoriamente para provar gradientes e recarga. Seu checkpoint é marcado `trained: false` e `engineering_smoke: true`; ele **não é** um modelo ENEM nem demonstra qualidade de correção. Nenhum treino completo de BERTimbau foi concluído nesta execução.

## Comparação e teste final

```bash
PYTHONPATH=backend .venv/bin/python -m app.ml.compare \
  --models artifacts/models/baseline artifacts/models/bertimbau-ordinal artifacts/models/bertimbau-regression \
  --output artifacts/reports/model_selection.json

PYTHONPATH=backend .venv/bin/python -m app.ml.evaluate \
  --model artifacts/models/baseline --split development \
  --output artifacts/reports/baseline_development.json

# Executar após fixar a seleção; não ajustar modelos usando este relatório.
PYTHONPATH=backend .venv/bin/python -m app.ml.evaluate \
  --model artifacts/models/baseline --split testing --final-test \
  --output artifacts/reports/baseline_final_test.json
```

`compare` exige a mesma proveniência de treino/validação e compara apenas as métricas já registradas de validação. Use apenas diretórios existentes e treinados; o exemplo com três candidatos depende de concluir ambos os treinamentos. O baseline desta execução teve sua seleção fixada no [registro de seleção](../artifacts/reports/baseline_selection.json) antes da avaliação final.

Os [relatórios](../artifacts/reports/) incluem, por competência e soma: MAE, RMSE, QWK quadrático, concordância exata, diferença absoluta até 40, matriz de confusão, distribuição de erro assinado e previsões por identificador. Métricas usam notas quantizadas; a regressão é limitada a 0..200 e arredondada ao múltiplo de 40 mais próximo, com empate para cima. QWK é `null` quando matematicamente indefinido. `macro.rmse` é a média dos cinco RMSE, não o RMSE da soma.

Resultado real do baseline `tfidf-ridge-5d3e145e9996`, alpha 1, escolhido entre 1/10/100:

| Split | N | MAE médio por competência | RMSE médio por competência | QWK médio |
|---|---:|---:|---:|---:|
| Validação | 686 | 22,0991 | 35,6137 | 0,4191 |
| Teste reservado | 686 | 21,1312 | 34,1778 | 0,4470 |

Esse acordo moderado e o corpus relativamente pequeno limitam o uso pedagógico. Distribuição de temas, fontes e anos pode diferir do uso atual. As notas são estimativas supervisionadas e devem ser revisadas por humanos. O resultado de teste já foi observado; futuros ajustes motivados por ele exigem um novo conjunto reservado para uma avaliação independente.

## Integração e validação

[Predictor](../backend/app/ml/predict.py) recebe o diretório do artefato. `predict(text)` retorna `scores` com `c1` a `c5` e `model_info` contendo versão, tipo e estado treinado. `status()` informa ausência/incompletude; `ModelUnavailableError` impede substituir ausência por notas fixas. A camada normativa e as justificativas usam a cartilha revisada, separadamente do preditor supervisionado.

## Treinamento otimizado e backups

O treinamento Transformer em [train.py](../backend/app/ml/train.py) possui dois
modos:

- `--training-mode full`: atualiza encoder e cabeça de pontuação.
- `--training-mode frozen`: congela o encoder em avaliação, desativa gradientes
  do encoder e treina somente a cabeça.

Os tokens são armazenados em cache content-addressed em
`<output>/token-cache`. A chave inclui tokenizer, textos/IDs, `max-length` e
`overlap`; alterações nesses elementos geram outro cache. O padding continua
dinâmico por batch e os tokens não são truncados silenciosamente.

Backups transacionais são controlados por [checkpoints.py](../backend/app/ml/checkpoints.py):

```bash
PYTHONPATH=backend .venv/bin/python -m app.ml.train \
  --data datasets/prepared \
  --base-model artifacts/models/bertimbau \
  --output artifacts/models/bertimbau-ordinal \
  --objective ordinal --training-mode frozen \
  --backup-interval-seconds 1800 --max-backups 2 \
  --backup-root artifacts/models/training-backups \
  --device cpu
```

Cada execução recebe um `run_id` e grava em
`artifacts/models/training-backups/<run_id>/`. O checkpoint só é renomeado
atomicamente após manifesto SHA-256 e validação dos arquivos. Falhas não
removem backups válidos. O contador não reutiliza números, inclusive após uma
gravação falha. Por padrão apenas os dois backups válidos mais recentes são
mantidos.

Para retomar um checkpoint de época:

```bash
PYTHONPATH=backend .venv/bin/python -m app.ml.train ... \
  --resume-latest --backup-root artifacts/models/training-backups \
  --run-id <run_id>
```

Também é possível usar `--resume-from <backup>/model`. A compatibilidade de
versão, arquitetura, dados e hiperparâmetros é verificada antes de carregar.
O estado do otimizador, scheduler, época, histórico e RNG é salvo. A retomada
também preserva a permutação da época, o próximo índice de batch e o `global_step`.
Assim, um backup criado após um `optimizer.step()` retoma no batch seguinte,
sem repetir a atualização confirmada. O backup é sempre um limite seguro; se o
processo cair durante o batch atual, esse batch ainda pode ser refeito, mas
nenhuma atualização parcialmente gravada é considerada.

Benchmark em textos reais do conjunto completo (32 redações, 229 segmentos,
janela 64, encoder frozen, duas passagens do batch representativo):

| Threads | Forward médio por batch | RSS máximo observado |
|---:|---:|---:|
| 4 | 2,370 s | 892,9 MB |
| 6 | 1,792 s | 894,6 MB |
| 8 | 1,544 s | 897,9 MB |
| 12 | 3,078 s | 899,2 MB |

O treino persistente foi iniciado com 8 threads, modo `frozen`, janela 256 e
sobreposição 32. O serviço systemd de usuário é `enem-bertimbau-frozen.service`;
seu log é `artifacts/models/training-logs/overnight.log` e os backups ficam em
`artifacts/models/training-backups/overnight-frozen/`. Consulte o estado com:

```bash
systemctl --user status enem-bertimbau-frozen.service
tail -f artifacts/models/training-logs/overnight.log
```

Pare com `systemctl --user stop enem-bertimbau-frozen.service`. Para retomar,
execute o mesmo comando com `--resume-latest --run-id overnight-frozen`; a
execução validará o manifesto e escolherá o backup mais recente íntegro.

```bash
PYTHONPATH=backend .venv/bin/python -m pytest \
  backend/tests/test_dataset.py backend/tests/test_metrics.py backend/tests/test_ml.py -q
```

Os testes verificam desserialização maliciosa, soma/escala de notas, preservação de parágrafos, quarentena entre splits, adulteração dos JSONL, métricas com resultado calculado manualmente, treino/recarga real do baseline, ausência explícita de artefato, bloqueio de teste sem flag, cobertura de todos os tokens, monotonicidade ordinal e atualização real de pesos/recarga de um Transformer pequeno.
