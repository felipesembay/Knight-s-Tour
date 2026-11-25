# Arquitetura Atual (as-is, 2026-11-24)

Retrato do sistema **antes** da evolução Human-in-the-Loop. Detalhes de problemas: `TECHNICAL_AUDIT.md`.

## Diretórios

```text
Knight-s-Tour/
├── index.html, script.js, styles.css   # Jogo (frontend estático)
├── start_game.py / start_game.sh       # Sobem Flask + http.server (portas divergentes, ver audit A9)
├── start_training.sh                   # Treino via tf_env (QUEBRADO: erro de sintaxe, audit A6)
├── models -> RL/models                 # symlink
├── docs/                               # Esta documentação
└── RL/
    ├── app.py            # Flask :5018 — /predict, /play_with_ai, /model_status
    ├── knight_env.py     # KnightTourEnv (gym.Env), estado (N,N,3), 8 ações
    ├── dqn_agent.py      # DQNAgent Double DQN Keras, replay deque
    ├── train.py          # Loop de treino (constantes no topo do arquivo)
    ├── model_config.py   # Seleção de "melhor modelo" por tamanho + fallbacks
    ├── models/5x5/       # 39 checkpoints *.weights.h5 (ativos)
    ├── logs/             # training_log_5x5.csv (anexado entre execuções)
    ├── resultados_passados/  # 1,5 GB de execuções antigas (5x5 e 6x6) — não versionado
    ├── scripts/          # testing/, training/, utils/, benchmark/
    └── docs/             # Docs antigas (COMO_TREINAR, COMO_JOGAR_COM_IA ...)
```

## Fluxo de jogo (modo clássico)

```text
Navegador (script.js)
  ├─ jogada local: valida movimento, atualiza tabuleiro, timer, leaderboard (localStorage)
  ├─ "Dica" (Warnsdorff local em JS)
  └─ "Dica da IA" (só 5x5) ──POST /predict {board:[[0|1|2]]}──► Flask app.py
                                                                  ├─ load_agent_for_size (cache por N)
                                                                  ├─ reconstrói KnightTourEnv a partir do board
                                                                  └─ agent.act(obs, mask) ► {action_index, move}
```

## Fluxo de treino

```text
train.py ─► KnightTourEnv.reset() (centro) ─► loop de passos:
   mask + onward_counts ─► agent.act (ε-greedy guiado por Warnsdorff)
   env.step ─► agent.remember ─► a cada 4 passos agent.replay(64)
fim de episódio ─► replay extra, target update a cada 5 ep., CSV, checkpoints (100 ep., vitória, best, final)
```

## Contratos implícitos

- Codificação do tabuleiro: `0` livre, `1` visitada, `2` cavalo (frontend, backend e env usam a mesma).
- Ordem das ações = `KNIGHT_MOVES` do spec (`[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]`). **Não alterar**: checkpoints dependem dela.
- Nome de checkpoint: `knight_tour_dqn_b{N}_e{ep}[_win|_best|_final].weights.h5`.
