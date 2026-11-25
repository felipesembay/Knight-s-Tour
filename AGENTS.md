# AGENTS.md — Guia para agentes de IA

Projeto: jogo Knight's Tour (Cavalo Solitário) com IA DQN, em evolução para treino Human-in-the-Loop.

## Comece por aqui
1. `docs/HANDOFF.md` — estado atual e próxima tarefa.
2. `docs/TECHNICAL_AUDIT.md` — problemas conhecidos e números de referência.
3. `docs/IMPLEMENTATION_PLAN.md` + `docs/TASKS.md` — o que fazer e em que ordem.

## Regras
- **Não reconstruir do zero.** Preservar modo clássico, endpoints v1 (`/predict`, `/play_with_ai`, `/model_status`) e o contrato do `script.js`.
- **Nunca apagar/sobrescrever checkpoints** em `RL/models/` ou `RL/resultados_passados/`. Modelos novos vão para `RL/models_v2/`.
- **Não alterar a ordem de `KNIGHT_MOVES`** (checkpoints dependem dela).
- Codificação do tabuleiro: 0 livre, 1 visitada, 2 cavalo.
- Não iniciar treinos longos sem pedido explícito. Benchmarks e testes devem rodar em minutos.
- Toda decisão técnica relevante → `docs/DECISIONS.md`. Todo experimento → `docs/EXPERIMENTS.md`.
- Ao terminar uma tarefa: atualizar `docs/TASKS.md`, `docs/PROGRESS.md`, `docs/HANDOFF.md`.
- Sem dados pessoais nos dados de experiência.

## Ambiente
- Python oficial: `~/miniconda3/envs/tf_env/bin/python` (3.8, TF 2.13.1, PyTorch 2.4 também presente).
- GPU (RTX 4060 8 GB):
  ```bash
  export LD_LIBRARY_PATH=~/miniconda3/envs/tf_env/lib/python3.8/site-packages/nvidia/cudnn/lib:~/miniconda3/envs/tf_env/lib:$LD_LIBRARY_PATH
  ```
  O Adam com XLA falha (`libdevice not found`) até T1.2 ser aplicada — use `jit_compile=False`.
- Forçar CPU: `CUDA_VISIBLE_DEVICES=-1`.

## Rodar
```bash
cd RL && ~/miniconda3/envs/tf_env/bin/python app.py      # API :5018
python -m http.server 8018                                # na raiz, frontend
cd RL && ~/miniconda3/envs/tf_env/bin/python train.py     # treino v1
```

## Mapa rápido
- `RL/knight_env.py` ambiente · `RL/dqn_agent.py` agente · `RL/train.py` treino · `RL/app.py` API · `RL/model_config.py` seleção de modelo · `script.js` frontend.
