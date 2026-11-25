# Esquema de Dados (proposta — aguarda seção 9 do briefing)

Banco: `RL/data/experience.db` (SQLite, `PRAGMA journal_mode=WAL`). Não versionado (`.gitignore`).

Codificação de estado: bitboard. Casa `(r,c)` → índice `r*N + c`. `visited` é INTEGER com bit `i` = 1 se a casa `i` foi visitada (inclui a casa atual). Até 9x9 = 81 bits → para N ≥ 8 (64+ bits) usar `visited_lo`/`visited_hi` (dois INTEGER de 63 bits úteis) ou BLOB de 11 bytes. Proposta: BLOB little-endian de `ceil(N²/8)` bytes (`visited`), uniforme para todos os N.

Máscara de ações: INTEGER 0–255, bit `a` = ação `a` válida (ordem `KNIGHT_MOVES`).

## Tabelas

```sql
CREATE TABLE schema_version (version INTEGER NOT NULL);

CREATE TABLE sessions (
  session_id     TEXT PRIMARY KEY,          -- uuid4
  mode           TEXT NOT NULL CHECK (mode IN ('classic','human_training','autonomous','collaborative')),
  source         TEXT NOT NULL CHECK (source IN ('human','agent','oracle','mixed')),
  board_size     INTEGER NOT NULL,
  start_pos      INTEGER NOT NULL,
  tour_type      TEXT NOT NULL DEFAULT 'open',
  status         TEXT NOT NULL CHECK (status IN ('in_progress','win','stuck','abandoned')),
  contribute     INTEGER NOT NULL DEFAULT 0, -- opt-in do usuário (RF2)
  moves_count    INTEGER NOT NULL DEFAULT 0,
  model_version  TEXT,                       -- modelo que sugeriu jogadas (colab)
  client_info    TEXT,                       -- JSON livre, sem dados pessoais
  created_at     TEXT NOT NULL,
  ended_at       TEXT
);

CREATE TABLE transitions (
  id             INTEGER PRIMARY KEY,
  session_id     TEXT NOT NULL REFERENCES sessions(session_id),
  t              INTEGER NOT NULL,           -- índice do passo na partida
  board_size     INTEGER NOT NULL,
  visited        BLOB NOT NULL,              -- estado s (bitboard)
  pos            INTEGER NOT NULL,
  mask           INTEGER NOT NULL,           -- ações válidas em s
  action         INTEGER NOT NULL,           -- ação executada
  reward         REAL NOT NULL,
  next_pos       INTEGER NOT NULL,           -- s' = visited | bit(next_pos)
  next_mask      INTEGER NOT NULL,
  done           INTEGER NOT NULL,
  source         TEXT NOT NULL,              -- human | agent | oracle | collab_intervention
  ai_action      INTEGER,                    -- sugestão da IA (colab)
  intervened     INTEGER NOT NULL DEFAULT 0,
  q_values       BLOB,                       -- 8 × float32, opcional
  oracle_label   TEXT CHECK (oracle_label IN ('good','fatal','unknown')),
  oracle_good    INTEGER,                    -- máscara de ações boas segundo o oráculo
  demo_weight    REAL NOT NULL DEFAULT 0,    -- peso p/ perda de margem (TRAINING_STRATEGY)
  reward_version INTEGER NOT NULL DEFAULT 1, -- permite recalcular recompensa
  created_at     TEXT NOT NULL,
  UNIQUE (session_id, t)
);
CREATE INDEX idx_tr_board_source ON transitions(board_size, source);

CREATE TABLE runs (
  run_id       TEXT PRIMARY KEY,
  config       TEXT NOT NULL,                -- JSON (episódios, N, device, ε, freq...)
  status       TEXT NOT NULL CHECK (status IN ('running','paused','stopped','finished','failed')),
  episodes_done INTEGER NOT NULL DEFAULT 0,
  updates_done  INTEGER NOT NULL DEFAULT 0,
  last_checkpoint TEXT,
  metrics      TEXT,                          -- JSON último resumo
  started_at   TEXT, updated_at TEXT
);

CREATE TABLE control (                        -- comandos Flask -> worker
  id INTEGER PRIMARY KEY, run_id TEXT, command TEXT CHECK (command IN ('pause','resume','stop')),
  created_at TEXT, consumed_at TEXT
);
```

## Regras

- `s'` é derivável: `next_visited = visited | (1 << next_pos)`; não é armazenado.
- Recompensa armazenada com `reward_version`; o learner pode recalcular a partir de (s, a, s', done) se a função mudar.
- Partida abandonada: `sessions.status='abandoned'`, transições mantidas, último `done=0` (não é terminal do MDP — não bootstrapar como terminal).
- Nada de dados pessoais. `client_info` não deve conter IP/e-mail.

## Tamanho estimado

~60 B/linha + índice → 1M transições ≈ 80–100 MB.
