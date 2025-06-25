#!/bin/bash

# Script para executar treinamento do Knight's Tour DQN
# Mantém a configuração antiga de plotagem no terminal

echo "=========================================="
echo "    TREINAMENTO KNIGHT'S TOUR DQN"
echo "=========================================="
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "../train.py" ]; then
    echo "❌ Erro: Execute este script do diretório RL/scripts/training/"
    echo "   cd RL/scripts/training && ./run_training.sh"
    exit 1
fi

# Verificar dependências
echo "🔍 Verificando dependências..."
python3 -c "import tensorflow, gym, numpy, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependências não encontradas. Instalando..."
    pip3 install -r ../../requirements.txt
fi

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p ../../models/6x6
mkdir -p ../../logs/6x6

# Verificar GPU
echo "🖥️  Verificando GPU..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs disponíveis: {len(gpus)}')
for gpu in gpus:
    print(f'  - {gpu}')
"

echo ""
echo "🚀 Iniciando treinamento..."
echo "   Pressione Ctrl+C para parar"
echo ""

# Executar treinamento
cd ../..
python3 train.py

echo ""
echo "✅ Treinamento concluído!"
echo "📊 Logs salvos em: logs/6x6/training_log.csv"
echo "💾 Modelos salvos em: models/6x6/" 