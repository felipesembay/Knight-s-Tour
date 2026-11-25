# HANDOFF

Atualizado: 2026-11-24 (Fase 1 parcial). Leia este arquivo primeiro; depois `AGENTS.md`, `TECHNICAL_AUDIT.md`, `IMPLEMENTATION_PLAN.md`.

## Estado atual
Fase 0 concluída. Fase 1 parcial: treino roda na GPU (T1.1–T1.4, T1.6, T1.8 feitos). Jogo funcionando com Dica da IA 5x5. Proprietário vai jogar antes da etapa de treino.

## Última tarefa concluída
Ajuste do treino para GPU + scripts de treino + jogo no ar (ver `PROGRESS.md`, 2026-11-24 Fase 1).

## Próxima tarefa
1. Receber seções 9+ do briefing.
2. T1.5 (portas em `start_game.py`/README) e T1.7 (`RL/tests/`).
3. Fase 2 (oráculo, BC/DQfD).

## Arquivos modificados / criados
- Fase 0: `docs/*.md`, `AGENTS.md`, `CLAUDE.md`, `RL/scripts/benchmark/*`.
- Fase 1: `RL/dqn_agent.py`, `RL/train.py`, `RL/model_config.py`, `start_training.sh`, `start_game.sh`.
- Antes destas etapas o proprietário já tinha mudanças não commitadas em `RL/dqn_agent.py`, `RL/knight_env.py`, `RL/train.py` (base sobre a qual a Fase 1 foi feita). Nada foi commitado.

## Treinar (scripts prontos)
```bash
./start_training.sh --smoke                          # 30 episódios, valida GPU (~5 s)
./start_training.sh                                  # 5x5, 3000 episódios, GPU
./start_training.sh --board-size 6 --episodes 5000   # 6x6
./start_training.sh --device cpu --batch-size 64
./start_training.sh --max-vram-mb 2048               # limita VRAM
nohup ./start_training.sh --board-size 6 --episodes 20000 > RL/logs/train_6x6.out 2>&1 &   # segundo plano
./start_training.sh --help
```
Saídas: `RL/models/{N}x{N}/run_<id>/` (`*_best_greedy`, `*_best`, `*_e<k>`, `*_win`, `*_final`) e `RL/logs/training_log_{N}x{N}_<id>.csv`. Para o jogo usar um modelo novo: copiar o `.weights.h5` para `RL/models/{N}x{N}/` e apontar `BEST_MODELS` em `RL/model_config.py`. Interromper: Ctrl+C (checkpoints periódicos já gravados; não há retomada ainda).

## Jogar
```bash
./start_game.sh     # usa .venv (Flask + TF 2.21); API :5018, jogo http://localhost:8018
```

## Decisões técnicas
Ver `DECISIONS.md` (ADR-001…007, todas "Propostas" — aguardam validação do proprietário/experimentos).

## Comandos executados
```bash
bash -n start_training.sh                                   # erro de sintaxe linha 68
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
~/miniconda3/envs/tf_env/bin/python RL/scripts/benchmark/bench.py {cpu|gpu} {5|6}
~/miniconda3/envs/tf_env/bin/python RL/scripts/benchmark/bench_nojit.py gpu {5|6}
~/miniconda3/envs/tf_env/bin/python RL/scripts/benchmark/bench_fused.py {cpu|gpu} 6 {64|256}
# avaliação gulosa dos checkpoints 5x5 e protótipo do oráculo: scripts ad hoc (resultados em EXPERIMENTS.md)
```
GPU precisa de: `export LD_LIBRARY_PATH=~/miniconda3/envs/tf_env/lib/python3.8/site-packages/nvidia/cudnn/lib:~/miniconda3/envs/tf_env/lib:$LD_LIBRARY_PATH`

## Testes realizados
Nenhum teste automatizado existe. Verificações manuais: benchmarks, carga de 39 checkpoints, avaliação gulosa.

## Resultados
Ver `TECHNICAL_AUDIT.md §1/§4` e `EXPERIMENTS.md`. Resumo: gargalo = overhead Keras (~100 ms/update, ~45× evitável); GPU quebrada no TF 2.13 (XLA); DQN 0% vitória gulosa; Warnsdorff/oráculo resolvem 6x6.

## Problemas conhecidos
Audit A1–A15. Resolvidos na Fase 1: A1 (train step), A2 (GPU), A3 (masking no alvo), A6 (`start_training.sh`), A7 (model_config). Abertos: A9 portas em `start_game.py`/README, A11 retenção, A12 retomada, A13 gym. Novo: `act()` por estado (~1 ms) agora domina o tempo do rollout.

## Experimentos em andamento
Nenhum treino rodando. Servidores do jogo podem estar ativos (`./start_game.sh`, Ctrl+C para parar).

## Checkpoints
- **Em uso no jogo** (`RL/model_config.py`): `RL/models/5x5/knight_tour_dqn_b5_run20261124_center_final.weights.h5` e `RL/models/6x6/knight_tour_dqn_b6_run20261124_center_final.weights.h5` (cópias dos `final` das runs 20261124_211422 / 211804; vencem do centro, generalizam pouco — EXP-008). Frontend: `AI_BOARD_SIZES = [5, 6, 7, 8]`, `index.html` carrega `script.js?v=<versão>` (mudar a versão a cada alteração do JS para evitar cache) e `AI_SERVER_URL` em `script.js`; antes da 1ª jogada a Dica da IA destaca a casa inicial (`starting_position` de `/model_status`).
- Runs: `RL/models/{N}x{N}/run_<id>/`. Nas runs 211422/211804 os `*_win` extras foram apagados a pedido do proprietário (ficou só o primeiro). Run `6x6/run_20261124_212736` (`--random-start`): 36/36 casas → **em uso no jogo** como `knight_tour_dqn_b6_run20261124_randomstart_best.weights.h5` (EXP-009). 7x7 em uso: `knight_tour_dqn_b7_run20261124_randomstart_best.weights.h5` (25/25, EXP-011). 8x8 em uso: `knight_tour_dqn_b8_run20261124_randomstart_best.weights.h5` (64/64; EXP-013). Nenhum treino em andamento.
- Gráficos: `RL/scripts/utils/plot_training_report.py` → `docs/images/` (usados no README).
- v1 antigos: `RL/models/5x5/*.weights.h5` (39; `e8_win` incompatível). Symlink `models -> RL/models`.
- v1 históricos: `RL/resultados_passados/models/{5x5,6x6}/` (1,5 GB, não versionado).
- v2: ainda não existe (`RL/models_v2/` planejado).

## Como retomar desenvolvimento
1. `cd RL && ~/miniconda3/envs/tf_env/bin/python -c "import tensorflow"` (ambiente oficial: `tf_env`).
2. Seguir `TASKS.md` na ordem; atualizar `PROGRESS.md`, `TASKS.md` e este arquivo ao fim de cada tarefa.

## Como retomar treino
Use `./start_training.sh` (seção "Treinar"). Não há retomada real: cada execução recomeça do zero (ε=1, buffer vazio). Retomada completa vem na Fase 2 (T2.4).
