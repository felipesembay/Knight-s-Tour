# Plano de Implementação

Ordem pensada para entregar valor medível cedo e nunca quebrar o modo clássico. Cada fase termina com testes verdes + docs atualizadas (`PROGRESS.md`, `HANDOFF.md`).

## Fase 0 — Auditoria ✅
Concluída (`TECHNICAL_AUDIT.md`).

## Fase 1 — Correções de base (baixo risco, alto ganho)
1. `start_training.sh`: corrigir erro de sintaxe (linhas duplicadas).
2. `dqn_agent.py`: `Adam(jit_compile=False)`; train step `tf.function`; masking no alvo (exige `next_mask` na transição — `remember()` ganha parâmetro opcional, retrocompatível).
3. `model_config.py`: ordenação de fallback robusta a `*_win`; apontar "best" para checkpoint existente medido (hoje `e293_win`, 24/25).
4. Unificar portas (5018/8018) em `start_game.py` e README.
5. Log CSV com `run_id`.
6. Criar `RL/tests/` + testes de masking e compatibilidade v1.

Saída: mesmo algoritmo, ~40× mais rápido, GPU funcional.

## Fase 2 — Núcleo v2
1. `RL/core/board.py` (bitboard) + testes de equivalência com `KnightTourEnv`.
2. `RL/core/oracle.py` (Warnsdorff, DFS com orçamento, paridade, cache).
3. `RL/core/features.py` (canais configuráveis).
4. `RL/hitl/learner.py`: rede, BC, DQfD (TD 1-step + n-step + margem), checkpoint completo (`tf.train.Checkpoint`: pesos, target, otimizador, ε, contadores, RNG).
5. Script `RL/train_v2.py` (CLI argparse) para BC → DQfD offline. EXP-002/003/004.

## Fase 3 — Persistência e treino autônomo controlável
1. `RL/hitl/store.py` (SQLite, `DATA_SCHEMA.md`) — **ajustar à seção 9 do briefing quando recebida**.
2. `RL/hitl/replay.py` (cache em RAM, amostragem por fonte).
3. `RL/hitl/worker.py` com pause/resume/stop via tabela `control`, limites de tempo/VRAM/threads, checkpoints periódicos, retomada.
4. Endpoints `/train/*`.

## Fase 4 — Modos humano e colaborativo
1. Endpoints `/session/*` + revalidação no servidor + rotulagem pelo oráculo.
2. `script.js`: seletor de modo, opt-in de contribuição, envio de jogadas, sugestão da IA com Q-values, registro de intervenção, abandono (`beforeunload` / reset → `/session/end`).
3. Aviso em N ímpar quando a casa inicial não tem tour.
4. `/predict` serve v2 quando promovido (ADR-004).

## Fase 5 — Painel e operação
1. UI de treino autônomo (status, controles).
2. Retenção de checkpoints; mover `.github/workflows` para a raiz.

## Riscos
- Seção 9+ do briefing pode alterar `DATA_SCHEMA` e a Fase 3.
- `gym` 0.26 + NumPy 2 no `.venv` (Python 3.12) — padronizar no `tf_env` ou migrar para `gymnasium`.
- Usuário mal-intencionado poluindo dados: mitigado por revalidação + oráculo.
