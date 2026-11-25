# Especificação da API (Flask, porta 5018)

## Existentes (v1) — manter compatíveis

### `POST /predict`
Req: `{"board": [[0|1|2, ...], ...]}` (0 livre, 1 visitada, 2 cavalo).
Resp 200: `{"action_index": int, "move": {"d_row": int, "d_col": int}, "model_info": {"board_size": "5x5", "model_file": str, "win_rate": float|null, "is_fallback": bool}}`
Resp 400: modelo indisponível / tabuleiro não quadrado. 500: erro interno.
Extensão proposta (retrocompatível): campos opcionais `q_values: [8 floats|null]`, `valid_mask: [8 bool]`, `model_version: "v1"|"v2:<run_id>"`.

### `POST /play_with_ai`
Req igual. Resp: `{"moves": [{action_index, d_row, d_col}], "success": bool, "visited_count": int, "total_squares": int, "model_info": {...}}`.

### `GET /model_status`
Resp: `{"5x5": {available, description, model_file, win_rate, starting_position, is_fallback}, ...}`.

## Novos (propostos)

### Sessões (modos treino humano e colaborativo)

| Método | Rota | Corpo | Resposta |
|---|---|---|---|
| POST | `/session/start` | `{mode, board_size, start_pos:[r,c], tour_type, contribute: bool}` | `{session_id}` |
| POST | `/session/move` | `{session_id, t, from:[r,c], to:[r,c], ai_action?: int, q_values?: [8], intervened?: bool}` | `{accepted: bool, reward, done, status, oracle_label?}` |
| POST | `/session/end` | `{session_id, status: "win"|"stuck"|"abandoned"}` | `{ok, transitions}` |
| POST | `/session/undo` | `{session_id}` | `{ok}` — marca a última transição como desfeita (não usada no treino) |

Regras: o servidor **revalida** cada movimento (não confia no cliente) e reconstrói o estado a partir do histórico da sessão. Se `contribute=false`, nada é persistido além de métricas agregadas anônimas (ou nada — decidir).

### Treino autônomo

| Método | Rota | Corpo / Resposta |
|---|---|---|
| POST | `/train/start` | `{board_size, episodes, start: "center"|"random_solvable"|[r,c], exploration: {type:"eps_warnsdorff"|"boltzmann", eps_start, eps_end}, device:"cpu"|"gpu", time_limit_s, train_every_steps, checkpoint_every_episodes, max_vram_mb, cpu_threads, speed: "max"|"throttled", use_human_data: bool}` → `{run_id}` |
| POST | `/train/pause` | `{run_id}` → `{ok}` |
| POST | `/train/resume` | `{run_id}` → `{ok}` |
| POST | `/train/stop` | `{run_id}` → `{ok}` (salva checkpoint antes de sair) |
| GET | `/train/status?run_id=` | `{status, episodes_done, updates_done, greedy_win_rate, avg_visited, epsilon, device, last_checkpoint, updated_at}` |

Erros: JSON `{"error": str}` com 400 (validação), 404 (sessão/run inexistente), 409 (run já ativo para o mesmo N), 500.
