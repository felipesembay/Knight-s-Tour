# Testes

## Estado atual
- Não existe pasta `tests/` (o `pyproject.toml` aponta para `tests/`). `RL/scripts/testing/` contém scripts manuais, não pytest.
- `RL/.github/workflows/ci.yml` existe mas está dentro de `RL/` → GitHub não o executa (precisa estar em `.github/` na raiz).

## Como rodar (proposto)
```bash
cd RL
~/miniconda3/envs/tf_env/bin/python -m pytest -q tests/        # unitários, CPU, < 60 s
~/miniconda3/envs/tf_env/bin/python -m pytest -q -m slow tests/ # integração (treino curto)
```
Forçar CPU em testes: `CUDA_VISIBLE_DEVICES=-1`.

## Plano de testes

| Área | Teste | Tipo |
|---|---|---|
| Motor bitboard | máscara de ações == `KnightTourEnv._get_valid_moves_mask()` em 10k estados aleatórios (5–9) | propriedade |
| Motor bitboard | observação 3 canais == `_get_observation()` (compatibilidade v1) | propriedade |
| Masking | `act()` nunca devolve ação inválida (ε=1 e ε=0) | unit |
| Masking | alvo Double DQN ignora ações inválidas; estado sem ações → bootstrap 0 | unit (batch sintético com Q inválido gigante) |
| Oráculo | todas as casas pares de 5x5 e todas de 6x6 resolvidas; paridade rejeita casas ímpares em N ímpar | unit |
| Store | round-trip sessão → transições → reconstrução do estado | unit |
| Store | abandono gera `status='abandoned'` e último `done=0` | unit |
| API | `/predict` resposta compatível com v1 (contrato do `script.js`) | integração (Flask test client) |
| API | `/session/move` rejeita movimento inválido / fora de ordem | integração |
| Retomada | treinar 200 updates, salvar, carregar, comparar pesos/ε/contadores/otimizador | integração |
| Pause/stop | worker obedece `control` em ≤ 1 lote | integração |
| Regressão | modelos v1 em `RL/models/5x5/` carregam (exceto `e8_win`, conhecido) | integração |
