# Arquitetura Alvo (to-be)

Evolução incremental de `CURRENT_ARCHITECTURE.md`. Decisões justificadas em `DECISIONS.md`.

## Visão geral

```text
┌──────────── Navegador (script.js) ─────────────┐
│ Modo: Clássico | Treino humano | Colaborativo  │
│ Painel de treino autônomo (start/pause/stop)   │
└──────┬──────────────────────────────┬──────────┘
       │ REST (Flask :5018)            │ polling /train/status
┌──────▼──────────────────────────────▼──────────┐
│ app.py (Flask)                                  │
│  ├─ /predict, /play_with_ai, /model_status (v1) │
│  ├─ /session/*  (registro de partidas)          │
│  └─ /train/*    (controle do worker)            │
└──────┬───────────────────────┬─────────────────┘
       │ grava                  │ comandos (tabela control)
┌──────▼──────────┐   lê   ┌───▼───────────────────────────┐
│ SQLite (WAL)    │◄───────┤ trainer worker (processo à    │
│ experience.db   │        │ parte, GPU/CPU)               │
│ sessions/moves/ │        │  ├─ rollouts autônomos         │
│ transitions/    │        │  ├─ oracle (Warnsdorff + DFS)  │
│ runs/control    │        │  ├─ learner DQfD (tf.function) │
└─────────────────┘        │  └─ checkpoints models_v2/     │
                           └────────────────────────────────┘
```

Flask **nunca treina** no thread da requisição: só grava experiências e comandos. O worker é um processo separado (`python -m RL.hitl.worker`), o que mantém a UI responsiva e permite limitar recursos.

## Módulos novos (propostos)

| Módulo | Responsabilidade |
|---|---|
| `RL/core/board.py` | Motor bitboard: `visited` (int), `pos`, máscara de ações, graus, obs para a rede. Sem dependência de gym. |
| `RL/core/features.py` | Construção de observação a partir do bitboard (canais configuráveis, ver `TRAINING_STRATEGY.md`). |
| `RL/core/oracle.py` | Warnsdorff, DFS com orçamento, poda de paridade, cache. |
| `RL/hitl/store.py` | Acesso ao SQLite (`DATA_SCHEMA.md`). |
| `RL/hitl/replay.py` | Buffer em RAM carregado do SQLite, amostragem por fonte/prioridade. |
| `RL/hitl/learner.py` | Rede, train step DQfD compilado, target network, save/load de estado completo. |
| `RL/hitl/worker.py` | Loop do treino autônomo + consumo de dados humanos + controle (pause/resume/stop). |
| `RL/hitl/api.py` | Blueprint Flask com `/session/*` e `/train/*` (`API_SPECIFICATION.md`). |

`knight_env.py`, `dqn_agent.py`, `train.py`, `model_config.py` permanecem compatíveis (v1). `KnightTourEnv` pode passar a delegar para `core/board.py` mantendo a mesma interface.

## Checkpoints (v2)

```text
RL/models_v2/{run_id}/
├── manifest.json        # arch_version, features, board_sizes, hiperparâmetros, métricas
├── latest/              # pesos online + target + otimizador + rng + contadores (tf.train.Checkpoint)
├── best.weights.h5      # melhor por avaliação gulosa
└── eval_history.jsonl
```

Retenção: `latest` + `best` + últimos K (default 3). Fim do acúmulo de 1,5 GB.

## Limites de recurso

- `tf.config.experimental.set_memory_growth` + limite opcional de VRAM (`set_logical_device_configuration`, ex. 2 GB).
- Threads TF configuráveis (`intra/inter_op_parallelism_threads`) para modo "segundo plano".
- Velocidade: `sleep` configurável entre lotes de rollout; limite de tempo por execução.
