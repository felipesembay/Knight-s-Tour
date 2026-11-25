#!/bin/bash
# Inicia o treino DQN do Knight's Tour usando o ambiente conda tf_env (GPU).
#
# Uso (da raiz do projeto):
#   ./start_training.sh                                   # padrões de RL/train.py (5x5, 3000 ep., GPU)
#   ./start_training.sh --board-size 6 --episodes 5000    # qualquer argumento de RL/train.py
#   ./start_training.sh --device cpu
#   ./start_training.sh --smoke                           # teste rápido (30 episódios) p/ validar GPU
#   ./start_training.sh --help                            # lista todos os argumentos
#
# Variáveis opcionais:
#   MINICONDA_DIR (padrão ~/miniconda3), ENV_NAME (padrão tf_env), PYTHON (força um interpretador)

set -euo pipefail

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-tf_env}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/RL"

if [ -n "${PYTHON:-}" ]; then
    ENV_PYTHON="$PYTHON"
elif [ -x "${MINICONDA_DIR}/envs/${ENV_NAME}/bin/python" ]; then
    ENV_PYTHON="${MINICONDA_DIR}/envs/${ENV_NAME}/bin/python"
else
    ENV_PYTHON="$(command -v python3 || true)"
fi
if [ -z "$ENV_PYTHON" ] || ! "$ENV_PYTHON" -c "import tensorflow" >/dev/null 2>&1; then
    echo "[ERRO] Python com TensorFlow não encontrado (tentado: ${ENV_PYTHON:-nenhum})."
    echo "       Crie o ambiente ${ENV_NAME} ou defina PYTHON=/caminho/python"
    exit 1
fi

# Bibliotecas CUDA/cuDNN instaladas via pip no tf_env (TF 2.13 precisa delas no LD_LIBRARY_PATH)
PYVER="$("$ENV_PYTHON" -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')"
ENV_PREFIX="$("$ENV_PYTHON" -c 'import sys; print(sys.prefix)')"
NVIDIA_DIR="${ENV_PREFIX}/lib/${PYVER}/site-packages/nvidia"
EXTRA_LIBS="${ENV_PREFIX}/lib"
if [ -d "$NVIDIA_DIR" ]; then
    for d in "$NVIDIA_DIR"/*/lib; do
        [ -d "$d" ] && EXTRA_LIBS="${d}:${EXTRA_LIBS}"
    done
fi
export LD_LIBRARY_PATH="${EXTRA_LIBS}:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

ARGS=()
for a in "$@"; do
    if [ "$a" = "--smoke" ]; then
        ARGS+=(--episodes 30 --checkpoint-every 10)
    else
        ARGS+=("$a")
    fi
done

echo "============================================="
echo "  Knight's Tour — Treinamento"
echo "============================================="
echo "Python : $ENV_PYTHON ($("$ENV_PYTHON" --version 2>&1))"
echo -n "GPU    : "
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponível"
GPU_COUNT="$("$ENV_PYTHON" -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo 0)"
echo "TF vê  : ${GPU_COUNT} GPU(s)"
if [ "$GPU_COUNT" = "0" ]; then
    echo "[AVISO] TensorFlow não encontrou GPU — treino vai rodar em CPU (ou falhar se --device gpu)."
fi
echo "Args   : ${ARGS[*]:-(padrões)}"
echo "============================================="
echo ""

cd "$PROJECT_DIR"
exec "$ENV_PYTHON" train.py ${ARGS[@]+"${ARGS[@]}"}
