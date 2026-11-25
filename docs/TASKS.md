# Tarefas

Status: `[ ]` a fazer · `[~]` em andamento · `[x]` feito · `[!]` bloqueado

## Fase 0 — Auditoria
- [x] T0.1 Ler README e docs existentes
- [x] T0.2 Mapear estrutura, dependências, ambientes (`tf_env`, `.venv`)
- [x] T0.3 Analisar rede, estado, ações, recompensa, replay, treino, checkpoints
- [x] T0.4 Analisar frontend ↔ backend
- [x] T0.5 Benchmark CPU/GPU curto (EXP-000)
- [x] T0.6 Avaliar checkpoints existentes (EXP-001)
- [x] T0.7 Protótipo de oráculo (EXP-001b)
- [x] T0.8 Documentação inicial (`docs/`, `AGENTS.md`, `CLAUDE.md`)
- [!] T0.9 Receber seções 9+ do briefing (texto chegou truncado)

## Fase 1 — Correções de base
- [x] T1.1 Corrigir sintaxe de `start_training.sh`
- [x] T1.2 `Adam(jit_compile=False)` em `dqn_agent.py`
- [x] T1.3 Train step `tf.function` + masking no alvo
- [x] T1.4 Fallback robusto em `model_config.py` + best 5x5 = `e293_win`
- [ ] T1.5 Unificar portas
- [x] T1.6 `run_id` no log CSV (arquivo e pasta de checkpoints por execução)
- [x] T1.8 Loader de pesos entre versões do Keras (feito: `_load_weights_by_shape`)
- [ ] T1.7 `RL/tests/` com testes de masking e compat v1

## Fase 2 — Núcleo v2
- [ ] T2.1 `core/board.py` + testes de equivalência
- [ ] T2.2 `core/oracle.py`
- [ ] T2.3 `core/features.py`
- [ ] T2.4 `hitl/learner.py` (BC, DQfD, checkpoint completo)
- [ ] T2.5 `train_v2.py` + EXP-002/003/004

## Fase 3 — Persistência / autônomo
- [!] T3.1 `hitl/store.py` (depende da seção 9)
- [ ] T3.2 `hitl/replay.py`
- [ ] T3.3 `hitl/worker.py` (pause/resume/stop, limites)
- [ ] T3.4 Endpoints `/train/*`

## Fase 4 — Humano / colaborativo
- [ ] T4.1 Endpoints `/session/*`
- [ ] T4.2 Frontend: seletor de modo + opt-in + registro
- [ ] T4.3 Colaborativo: sugestão + Q-values + intervenção
- [ ] T4.4 DAgger com oráculo
- [ ] T4.5 Aviso de paridade em N ímpar

## Fase 5 — Operação
- [ ] T5.1 Painel de treino
- [ ] T5.2 Retenção de checkpoints
- [ ] T5.3 CI na raiz
