# Auditoria Técnica — Knight's Tour (Fase 4)

Data: 2026-11-24 · Escopo: leitura de código + benchmark curto (< 5 min por medição). Nenhum código de produção foi alterado durante a auditoria.

## 1. Resumo executivo

| # | Achado | Severidade | Evidência |
|---|--------|-----------|-----------|
| A1 | `replay()` custa ~100 ms por batch de 64 — 3× `model.predict` + `model.fit` por chamada (overhead Keras, não computação). | **Crítico** (gargalo nº 1) | Benchmark §4 |
| A2 | Treino em GPU **quebra** no TF 2.13 do `tf_env`: `libdevice not found at ./libdevice.10.bc` (Adam com XLA). Treino só roda em CPU, ou com `jit_compile=False` / `XLA_FLAGS`. | **Crítico** | §4 |
| A3 | Alvo Double DQN **não aplica action masking**: `argmax` em `next_q_main` considera ações inválidas no próximo estado. | Alto (correção) | `RL/dqn_agent.py` `replay()` |
| A4 | Nenhum checkpoint 5x5 vence com política gulosa. Melhor: 24/25 (`e293_win`). Arquivos `*_win` registram vitórias **com exploração**, não da política. | Alto | §5 |
| A5 | Heurística de Warnsdorff (desempate aleatório) vence 100% no 5x5 e 80,5% no 6x6 a partir do centro — muito acima do DQN (0%). Excelente especialista para demonstrações/DAgger. | Oportunidade | §4 |
| A6 | `start_training.sh` tem erro de sintaxe (linhas duplicadas de merge manual; `if` sem `fi`). Não executa. | Alto | `bash -n` → erro linha 68 |
| A7 | `model_config.py`: modelo "melhor" (`e9100`) não existe; fallback ordena por episódio com `int()` que falha em `*_win` → `except` devolve ordem arbitrária do `glob` (hoje `e1600`). | Médio | §5 |
| A8 | `knight_tour_dqn_b5_e8_win.weights.h5` incompatível com a arquitetura atual (0 variáveis). | Baixo | §5 |
| A9 | Portas inconsistentes: `app.py`/`script.js`/`start_game.sh` = 5018/8018; `start_game.py` e README = 5001/8000. | Médio | grep |
| A10 | Log CSV é anexado entre execuções sem identificador de execução (episódios reiniciam em 0 no mesmo arquivo: 7342 linhas em `logs/training_log_5x5.csv`). | Médio | |
| A11 | Checkpoints a cada 100 episódios + a cada vitória → 155 arquivos / 1,5 GB em `RL/resultados_passados`. Sem retenção. | Médio | `du` |
| A12 | Sem retomada: `epsilon`, otimizador, replay buffer e contador de episódios não são persistidos. | Alto (requisito) | `train.py` |
| A13 | `gym` 0.26 (sem manutenção) usado com API antiga (`reset()` → obs, `step()` → 4-tupla). | Baixo | |
| A14 | Penalidade de movimento inválido (-5) é código morto: `act()` sempre mascara. | Baixo | |
| A15 | Frontend só habilita "Dica da IA" em 5x5 (`script.js:358`, `:452`). | Info | |

## 2. Stack e ambientes

- Frontend: HTML/CSS/JS puro (`index.html`, `script.js`, `styles.css`), servido por `python -m http.server`.
- Backend: Flask + flask-cors (`RL/app.py`), porta 5018. Endpoints `/predict`, `/play_with_ai`, `/model_status`.
- ML: TensorFlow/Keras (`RL/dqn_agent.py`), ambiente Gym (`RL/knight_env.py`), treino `RL/train.py`.
- Ambientes Python encontrados:
  - `~/miniconda3/envs/tf_env`: Python 3.8.20, TF 2.13.1, NumPy 1.24.3, gym 0.26.2, **PyTorch 2.4.1+cu121 (CUDA disponível)**. É o ambiente usado por `start_training.sh`.
  - `.venv` (raiz): Python 3.12, TF 2.21, NumPy 2.4 — sem verificação de GPU; gym 0.26 não suporta NumPy 2 oficialmente.
- Hardware: RTX 4060 8 GB (driver 580), 16 threads CPU, 31 GB RAM (~9 GB livres no momento da medição).

## 3. Arquitetura atual (ML)

- **Estado**: tensor `(N, N, 3)` float32 — canais: posição do cavalo, casas visitadas (excluindo a atual), casas alcançáveis válidas. Não inclui dimensão do tabuleiro, progresso nem graus.
- **Ações**: 8 (movimentos do cavalo, ordem `KNIGHT_MOVES` do spec — idêntica à do código).
- **Rede**: Conv2D(32,3×3) → Conv2D(64,3×3) → Flatten → Dense(64) → Dense(8). 122k parâmetros (5x5), 167k (6x6). Flatten amarra o modelo ao tamanho do tabuleiro → um modelo por N.
- **Recompensa**: +0,1 por passo + bônus Warnsdorff `max(0, 5-onward)*0,05`; vitória +50; travado `-(10 + restantes)`; inválido -5 (inalcançável).
- **Exploração**: ε-greedy com amostragem ponderada por Warnsdorff (`1/(onward+1)^2`) dentro das ações válidas.
- **Replay**: `deque(maxlen=100000)` de tuplas com arrays float32 completos (estado + próximo estado). Em memória; perdido ao fim do processo.
- **Treino**: replay a cada 4 passos + 1 ao fim do episódio; target update a cada 5 episódios; ε decai por replay (não por episódio).
- **Persistência**: somente pesos (`save_weights`). Sem estado do otimizador, ε, buffer, RNG.

## 4. Benchmark inicial (controlado)

Scripts: `RL/scripts/benchmark/bench.py`, `bench_nojit.py`, `bench_fused.py` (como rodar: `docs/EXPERIMENTS.md` EXP-000). Início no centro.

| Métrica | CPU 5x5 | CPU 6x6 | GPU 5x5 | GPU 6x6 |
|---|---|---|---|---|
| Passos de ambiente/s (com máscara + contagem Warnsdorff) | 9 740 | 8 749 | 9 626 | 8 798 |
| `act()` latência (1 estado) | 1,11 ms | 1,17 ms | 1,57 ms | 1,67 ms |
| `replay(64)` atual | 95 ms | 105 ms | **crash** (XLA) / 114 ms com `jit_compile=False` | 117 ms |
| Train step compilado (`tf.function`, mascarado), batch 64 | — | 2,78 ms | — | 2,26 ms |
| Idem, batch 256 | — | 7,32 ms | — | 2,33 ms |
| Comprimento médio, política aleatória válida | 13,3 | 18,9 | | |
| Warnsdorff (desempate aleatório), taxa de vitória (200 ep.) | **100%** | **80,5%** | | |

Conclusões:
1. **O gargalo não é a GPU.** É overhead de chamada Keras em `replay()` (~40× acima do custo real). Um train step em `tf.function` reduz de ~105 ms para ~2,3 ms.
2. Com train step compilado, GPU vale a pena com batch ≥ 256 (mesmo custo de 64); CPU escala linear com o batch.
3. `act()` por estado único (~1 ms) domina o rollout depois da correção: usar `tf.function` para inferência ou rodar vários ambientes vetorizados.
4. Ambiente (~9k passos/s em NumPy Python) não é gargalo agora; vira gargalo depois de A1 corrigido → bitboards/NumPy vetorizado.
5. Eficiência amostral é o problema real no 6x6: recompensa esparsa (+50 só no fim), DQN puro nunca venceu em ~5 000 episódios (`resultados_passados/logs/training_log_6x6.csv`: média 20–21/36 casas). Demonstrações de Warnsdorff (80% de vitória) devem ser o atalho principal.

## 5. Compatibilidade de modelos

- 39 checkpoints em `RL/models/5x5/` (58 MB); 38 carregam na arquitetura atual; `e8_win` falha.
- Avaliação gulosa do centro: 0 vitórias; melhor 24/25 (`e293_win`), depois 22 (`e1500`).
- `model_config.get_model_info(5)` resolve para `e1600` (fallback por ordem arbitrária, A7).
- 6x6 antigos em `RL/resultados_passados/models/6x6/` (mesmo formato `.weights.h5`).
- Estratégia de migração: ver `docs/DECISIONS.md` (ADR-004). Modelos antigos são preservados e continuam servidos pelo modo clássico enquanto o novo formato não existir para o tamanho.

## 6. Desperdícios por episódio

- `_get_observation()` recalcula máscara e aloca 3 arrays a cada passo; `train.py` chama `_get_valid_moves_mask()` e `get_action_onward_counts()` de novo → máscara calculada ~3× por passo.
- Estados armazenados duas vezes (s e s') como float32 `(N,N,3)`: 6x6 → 864 B por estado; 100k transições ≈ 173 MB. Bitboard (`uint64` visitadas + posição) → ~10 B por estado.
- Log CSV aberto/fechado a cada episódio (custo baixo, mas sem buffer).
- `update_target_model()` via `get_weights/set_weights` (cópia por NumPy) — aceitável a cada 5 episódios.
- `app.py` reconstrói `KnightTourEnv` a cada requisição (chama `reset()` no `__init__`) — barato, sem vazamento.

## 7. Memória / vazamentos

- Nenhum vazamento identificado no loop. `deque` limitado. `model.predict` em loop pode acumular grafos em versões antigas do TF; não observado no benchmark curto.
- `app.py` roda `debug=True` (reloader → dois processos carregando TF). Não usar em produção.

## 8. Frontend ↔ backend

- `script.js` envia `{"board": [[0|1|2]]}` (0 livre, 1 visitada, 2 cavalo) para `POST http://localhost:5018/predict`; recebe `{action_index, move:{d_row,d_col}, model_info}`.
- `/play_with_ai` existe no backend mas não é usado pelo frontend atual.
- Não há sessão, identificação de partida, nem registro de jogadas no backend.
