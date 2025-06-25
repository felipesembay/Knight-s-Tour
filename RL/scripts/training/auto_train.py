#!/usr/bin/env python3
"""
Script Automático de Treinamento - Knight's Tour DQN
Mantém a configuração antiga de plotagem no terminal
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    required_packages = ['tensorflow', 'gym', 'numpy', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Dependências faltando: {', '.join(missing_packages)}")
        print("📦 Instalando dependências...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', '../../requirements.txt'])
        return False
    return True

def setup_directories():
    """Cria diretórios necessários"""
    directories = ['../../models/6x6', '../../logs/6x6']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Diretório criado/verificado: {directory}")

def check_gpu():
    """Verifica configuração da GPU"""
    try:
        import tensorflow as tf
        print("🖥️  Configuração da GPU:")
        print(f"   TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   GPUs disponíveis: {len(gpus)}")
        for gpu in gpus:
            print(f"     - {gpu}")
        return len(gpus) > 0
    except Exception as e:
        print(f"⚠️  Erro ao verificar GPU: {e}")
        return False

def run_training(board_size=6, episodes=10000):
    """Executa o treinamento"""
    print("=" * 60)
    print(f"🚀 INICIANDO TREINAMENTO - Tabuleiro {board_size}x{board_size}")
    print(f"📊 Episódios: {episodes}")
    print("=" * 60)
    print("   Pressione Ctrl+C para parar")
    print("")
    
    try:
        # Mudar para o diretório RL e executar o script de treinamento
        os.chdir('../..')
        subprocess.run([sys.executable, 'train.py'])
        print("\n✅ Treinamento concluído com sucesso!")
        return True
    except KeyboardInterrupt:
        print("\n⏹️  Treinamento interrompido pelo usuário")
        return False
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Treinamento Automático - Knight\'s Tour DQN')
    parser.add_argument('--board-size', type=int, default=6, help='Tamanho do tabuleiro (padrão: 6)')
    parser.add_argument('--episodes', type=int, default=10000, help='Número de episódios (padrão: 10000)')
    parser.add_argument('--skip-checks', action='store_true', help='Pular verificações de dependências')
    
    args = parser.parse_args()
    
    print("==========================================")
    print("    TREINAMENTO KNIGHT'S TOUR DQN")
    print("==========================================")
    print("")
    
    # Verificar se estamos no diretório correto
    if not Path('../../train.py').exists():
        print("❌ Erro: Execute este script do diretório RL/scripts/training/")
        print("   cd RL/scripts/training && python3 auto_train.py")
        sys.exit(1)
    
    # Verificações (se não for pulado)
    if not args.skip_checks:
        print("🔍 Verificando dependências...")
        if not check_dependencies():
            print("❌ Falha ao instalar dependências")
            sys.exit(1)
        
        print("📁 Configurando diretórios...")
        setup_directories()
        
        print("🖥️  Verificando GPU...")
        gpu_available = check_gpu()
        if not gpu_available:
            print("⚠️  GPU não detectada - treinamento será executado na CPU")
    
    # Executar treinamento
    success = run_training(args.board_size, args.episodes)
    
    if success:
        print("\n📊 Resultados:")
        print("   - Logs: logs/6x6/training_log.csv")
        print("   - Modelos: models/6x6/")
        print("   - Último modelo: models/6x6/knight_tour_dqn_b6_final.h5")
    else:
        print("\n❌ Treinamento não foi concluído com sucesso")

if __name__ == "__main__":
    main() 