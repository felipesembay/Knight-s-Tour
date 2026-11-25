# Registro de Decisões (ADR)

Formato: contexto → decisão → consequências. Status: Proposta | Aceita | Substituída.

## ADR-001 — Manter TensorFlow/Keras como framework (Proposta)
- Contexto: `tf_env` tem TF 2.13 e PyTorch 2.4 (CUDA OK). Todo checkpoint existente é Keras `.weights.h5`.
- Decisão: continuar em TF/Keras; substituir `predict`+`fit` por train step em `tf.function`.
- Consequências: modelos antigos continuam carregáveis; ganho medido ~45× no train step (audit §4). Migração para PyTorch só se aparecer bloqueio real.

## ADR-002 — Corrigir GPU com `jit_compile=False` no Adam (Proposta)
- Contexto: Adam com XLA falha (`libdevice not found`). `libdevice.10.bc` existe em `~/miniconda3/envs/tf_env/lib/`, mas XLA procura em `<dir>/nvvm/libdevice/`.
- Decisão: `Adam(..., jit_compile=False)` no código (portável). Alternativa documentada: `XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit`.
- Consequências: GPU funcional; sem dependência de layout de diretórios CUDA.

## ADR-003 — Action masking nos alvos Double DQN (Proposta)
- Decisão: armazenar `next_mask` (uint8, 8 bits) por transição; `argmax` do online só entre ações válidas; terminal ou sem ações válidas → bootstrap 0.
- Consequências: muda o formato da transição (já muda com o buffer persistente).

## ADR-004 — Estratégia de migração de modelos (Proposta)
- Modelos antigos (`(N,N,3)` + Flatten) ficam intocados em `RL/models/{N}x{N}/` e continuam servidos por `model_config.py` no modo clássico.
- Modelos novos vão para `RL/models_v2/{run_id}/` com `manifest.json` (versão de arquitetura, features, N, métricas).
- `/predict` escolhe v2 se houver modelo v2 promovido para o N; senão v1. Nunca sobrescrever v1.
- Inicialização de v2 a partir de v1: não (features diferentes). v2 é pré-treinado por demonstrações (ADR-006), o que é mais rápido que reaproveitar pesos de 0% de vitória.

## ADR-005 — Buffer persistente em SQLite + estados compactos (Proposta, aguarda seção 9 do briefing)
- Contexto: estados float32 `(N,N,3)` ≈ 864 B (6x6), duplicados em s/s'. Precisamos de metadados (origem, sessão, qualidade, resultado) e escrita concorrente Flask × treinador.
- Decisão: SQLite (WAL) com estado em bitboard (`visited` BLOB de `ceil(N²/8)` bytes — 9x9 tem 81 casas, não cabe em INTEGER 64 bits; `pos` INTEGER). Observação reconstruída sob demanda (barato). Ver `DATA_SCHEMA.md`.
- Consequências: ~40 B/transição; 1M transições ≈ 40 MB. Buffer de treino em RAM é cache carregado do SQLite.

## ADR-006 — Especialista artificial: Warnsdorff + backtracking (Proposta)
- Contexto: Warnsdorff vence 100% (5x5) / 80,5% (6x6) do centro; DQN atual 0%. DFS com ordenação de Warnsdorff resolve 5x5–8x8 em milissegundos na maioria dos estados.
- Decisão: oráculo `solve(state)` → conjunto de ações que mantêm o tour completável. Usado para (a) demonstrações sintéticas, (b) rótulos DAgger, (c) pontuar qualidade de jogadas humanas, (d) métrica de avaliação.
- Medição (protótipo Python puro, DFS + ordenação Warnsdorff, orçamento 200k nós, todas as casas iniciais):
  - 6x6: 36/36 resolvidas, mediana 0,16 ms, máx 0,7 ms. 8x8: 64/64, mediana 0,33 ms, máx 180 ms.
  - 5x5: 13 resolvidas = exatamente as casas com `(r+c)` par; 12 estouram orçamento (sem tour — paridade de cores em N ímpar). 7x7: 22/25 casas pares resolvidas, 27 estouros (24 ímpares sem solução + 3 pares).
- Consequências: elimina a necessidade de rotulagem humana para DAgger. Exige (1) poda por paridade em N ímpar (casa de início/estado com mais casas livres da cor "errada" é impossível), (2) orçamento de nós com resultado `unknown`, (3) cache por `(N, visited, pos)`.
- Implicação de produto: em 5x5/7x7/9x9, iniciar numa casa ímpar torna o tour impossível. `random_start` no treino e o frontend devem tratar isso (avisar o usuário; nunca penalizar o agente por estado inviável).

## ADR-007 — Rede totalmente convolucional, independente de N (Proposta, condicionada a EXP-003)
- Ideia: CNN sem Flatten produz mapa de scores `(N,N)`; Q(a) = score na casa destino da ação `a` (+ cabeça de valor por pooling global). Um único modelo serve 5x5…9x9 e permite currículo 5→6→7.
- Adotar só se EXP-003 mostrar ≥ desempenho do modelo por-N no 6x6.
