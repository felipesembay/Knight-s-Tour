# Experimentos

Regra: todo experimento registra hipótese, comando, config, hardware, duração, resultado e conclusão. Métrica principal: **taxa de vitória gulosa** sobre todas as casas iniciais solúveis (e separadamente do centro).

## EXP-000 — Benchmark de referência (concluído, 2026-11-24)

- Hipótese: GPU é o gargalo do treino.
- Comandos (a partir da raiz do repo):
  ```bash
  PY=~/miniconda3/envs/tf_env/bin/python
  export LD_LIBRARY_PATH=~/miniconda3/envs/tf_env/lib/python3.8/site-packages/nvidia/cudnn/lib:~/miniconda3/envs/tf_env/lib:$LD_LIBRARY_PATH
  $PY RL/scripts/benchmark/bench.py cpu 6          # código atual, CPU
  $PY RL/scripts/benchmark/bench_nojit.py gpu 6    # código atual com Adam jit_compile=False (GPU)
  $PY RL/scripts/benchmark/bench_fused.py gpu 6 256  # train step tf.function mascarado
  ```
- Resultado: ver `TECHNICAL_AUDIT.md §4`. `replay(64)` ≈ 100 ms (CPU e GPU); train step compilado 2,3 ms (GPU, batch 64 ou 256). `bench.py gpu` com código atual **falha** (XLA libdevice).
- Conclusão: hipótese rejeitada. Gargalo = overhead Keras por chamada + eficiência amostral.

## EXP-001 — Avaliação gulosa dos checkpoints 5x5 existentes (concluído)
- 39 arquivos, centro: 0 vitórias; melhor `e293_win` 24/25; `e8_win` não carrega.

## EXP-001b — Oráculo DFS + Warnsdorff (concluído, protótipo)
- 6x6: 36/36 inícios resolvidos, mediana 0,16 ms. 5x5: só casas pares têm solução. Ver ADR-006.

## Planejados

| ID | Hipótese | Critério de sucesso |
|---|---|---|
| EXP-002 | Canais de grau + progresso melhoram BC/DQfD vs 3 canais | +10 p.p. vitória gulosa 6x6 com mesmo orçamento |
| EXP-003 | Rede totalmente convolucional ≥ rede por-N | Vitória 6x6 ≥ baseline; e generaliza para 7x7 sem re-treino > 0% |
| EXP-004 | BC puro do oráculo resolve 6x6 | ≥ 90% vitória gulosa (todas as casas) em ≤ 15 min GPU |
| EXP-005 | DQfD supera BC em robustez (estados fora da distribuição) | Melhor vitória a partir de estados de meio de jogo gerados por política aleatória |
| EXP-006 | n-step (n=5) acelera DQN puro 6x6 | Primeira vitória gulosa em menos episódios que 1-step |
| EXP-007 | Dados humanos (filtrados pelo oráculo) não degradam | Vitória não cai > 2 p.p. ao misturar 20% de dados humanos |

## EXP-008 — DQN v1 corrigido (Fase 1), início no centro (concluído, 2026-11-24)
- Execução do proprietário: `./start_training.sh` (5x5, 5000 ep., 3,2 min) e `--board-size 6` (5000 ep., 4,2 min), GPU.
- Runs: `RL/models/5x5/run_20261124_211422/`, `RL/models/6x6/run_20261124_211804/`.
- Curva (vitória no treino, com ε): 5x5 0,26 → 0,98; 6x6 0,02 → 0,97 (ε mínimo a partir de ~ep. 2500).
- Avaliação gulosa (scratchpad `eval_all.py`):

| Modelo | Centro | Vitórias, todas as casas com solução | Média casas |
|---|---|---|---|
| 5x5 best_greedy | 25/25 | 2/13 | 19,1 |
| 5x5 final | 25/25 | 4/13 | 20,3 |
| 6x6 best_greedy | 36/36 | 5/36 | 31,1 |
| 6x6 final | 36/36 | 16/36 | 31,9 |

- Conclusão: as correções da Fase 1 (masking no alvo, ε calibrado, throughput) bastaram para o DQN resolver 5x5 e 6x6 **a partir do centro** — antes, 0% de vitória e 6x6 nunca resolvido. A generalização para outras casas iniciais é fraca (treino só no centro) → próximo passo: `--random-start` (só casas com solução) e/ou abordagem híbrida.
- Problema: gravar `*_win` a cada vitória gerou 7 260 arquivos / 4,2 GB. Corrigido em `train.py`: só a primeira vitória é gravada.

## EXP-009 — 6x6 com `--random-start` (concluído, 2026-11-24)
- `./start_training.sh --board-size 6 --episodes 10000 --random-start` (8,7 min, GPU). Run `RL/models/6x6/run_20261124_212736/`.
- Guloso: `best_greedy` e `final` completam o tour a partir das **36/36** casas iniciais (vs 16/36 do modelo treinado só no centro).
- Promovido ao jogo: `RL/models/6x6/knight_tour_dqn_b6_run20261124_randomstart_best.weights.h5`.
- Correção posterior: `--random-start` agora sorteia só casas com solução (`KnightTourEnv.solvable_starts()`; em N ímpar só `(r+c)` par). Antes, em 5x5/7x7 sorteava casas sem passeio possível.

## EXP-010 — 7x7, início no centro (concluído, 2026-11-24)
- `./start_training.sh --board-size 7 --episodes 5000` (5,0 min, GPU). Run `RL/models/7x7/run_20261124_213819/`.
- Guloso: `best_greedy` centro 49/49, 2/25 casas com solução; `final` centro 49/49, **4/25**, média 42,0.
- Promovido: `RL/models/7x7/knight_tour_dqn_b7_run20261124_center_final.weights.h5`. Próximo: `--random-start` (agora só casas com solução).

## EXP-011 — 7x7 com `--random-start` (concluído, 2026-11-24)
- 20 000 episódios, ~21 min (estimado por mtime). Run `RL/models/7x7/run_20261124_214523/`.
- Guloso: `best_greedy` **25/25** casas com solução; `final` 23/25. Curva de treino oscila (queda a 0% perto do ep. 7000, recupera; termina em ~83% com ε=0,01).
- Promovido: `RL/models/7x7/knight_tour_dqn_b7_run20261124_randomstart_best.weights.h5`.

## Relatório gráfico
- `RL/scripts/utils/plot_training_report.py` → `docs/images/{curvas_aprendizado,antes_depois_6x6,mapa_casas_iniciais,resumo_generalizacao}.png` + `avaliacao.json`. Para incluir nova execução, acrescentar em `RUNS`.

## EXP-012 — 8x8, início no centro (concluído, 2026-11-24)
- `./start_training.sh --board-size 8 --episodes 10000` (12,5 min, GPU). Run `RL/models/8x8/run_20261124_223740/`.
- Curva: vitórias só a partir de ~ep. 5500; salto para ~90% em ~6200; termina em ~84% (ε=0,01).
- Guloso: `best_greedy` centro (4,4) 64/64, 1/64 casas; `final` centro 64/64, **9/64**, média 55,8.
- Promovido: `RL/models/8x8/knight_tour_dqn_b8_run20261124_center_final.weights.h5`. Em andamento (proprietário): `--random-start`, 30 000 ep., run 20261124_225056.

## EXP-013 — 8x8 com `--random-start` (concluído, 2026-11-24)
- 30 000 episódios, 47,4 min (GPU). Run `RL/models/8x8/run_20261124_225056/`.
- Guloso: `best_greedy` **64/64** casas; `final` 61/64.
- Promovido: `RL/models/8x8/knight_tour_dqn_b8_run20261124_randomstart_best.weights.h5`.
- Com isso, 6x6, 7x7 e 8x8 vencem de todas as casas com solução usando só DQN corrigido (sem a abordagem híbrida). 5x5 ainda só treinado no centro.
