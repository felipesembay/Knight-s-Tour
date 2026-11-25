# Progresso

## 2026-11-24 — Fase 0 (auditoria) concluída
- Auditoria completa: `TECHNICAL_AUDIT.md` (15 achados).
- Benchmarks EXP-000, EXP-001, EXP-001b executados (todos < 5 min, nenhum treino longo).
- Scripts de benchmark adicionados em `RL/scripts/benchmark/` (novos arquivos; nenhum arquivo existente alterado).
- Documentação criada: 14 arquivos em `docs/` + `AGENTS.md` + `CLAUDE.md`.
- Bloqueio: briefing truncado na seção 9. Fases 1–2 podem seguir; Fase 3 (store) aguarda.

Principais números:
- `replay(64)`: ~100 ms → train step compilado ~2,3 ms (GPU).
- GPU no código atual: falha (XLA libdevice).
- DQN atual: 0% vitória gulosa 5x5; Warnsdorff: 100% (5x5), 80,5% (6x6); oráculo DFS: 100% 6x6 em < 1 ms.

## 2026-11-24 — Fase 1 parcial: treino na GPU
- `RL/dqn_agent.py`: Adam sem XLA (corrige `libdevice not found`), train step `tf.function`, buffer circular pré-alocado, masking no alvo Double DQN (`next_mask`), inferência compilada, `q_values()`, loader de pesos por formato (Keras 2.13 ↔ Keras 3). Arquitetura da rede inalterada.
- `RL/train.py`: CLI (`--board-size --episodes --batch-size --device --max-vram-mb --checkpoint-every --random-start --seed ...`), cada execução em `models/{N}x{N}/run_<id>/` + `logs/training_log_{N}x{N}_<id>.csv` (nunca sobrescreve checkpoints antigos), avaliação gulosa em cada checkpoint (`*_best_greedy.weights.h5`), resumo correto para janelas < 100.
- **Mudança de comportamento**: `EPSILON_DECAY` agora considera que ε decai por update (não por episódio). Antes ε chegava ao mínimo em ~15% do treino, contrariando o comentário "70% do treino".
- `start_training.sh` reescrito (sintaxe OK, repassa argumentos, `--smoke`).
- `RL/model_config.py`: melhor 5x5 = `e293_win` (24/25 guloso); ordenação de fallback corrigida.
- `start_game.sh`: mensagem de modelo corrigida.
- Verificado: smoke test 30 episódios na GPU (RTX 4060) ~30 ep/s; `/predict` e `/play_with_ai` respondem no `.venv` com Q-values idênticos aos do `tf_env`.
- Achado: antes desta etapa o jogo (`.venv`, Keras 3) **não conseguia carregar nenhum modelo** — Dica da IA sempre falhava.
