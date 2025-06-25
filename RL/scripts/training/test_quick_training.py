#!/usr/bin/env python3
"""
Teste Rápido de Treinamento - Knight's Tour DQN
Executa apenas 10 episódios para verificar se tudo está funcionando
"""

import os
import sys
import subprocess
from pathlib import Path

def test_training():
    """Testa o treinamento com poucos episódios"""
    print("🧪 TESTE RÁPIDO DE TREINAMENTO")
    print("=" * 40)
    
    # Verificar se estamos no diretório correto
    if not Path('../../train.py').exists():
        print("❌ Erro: Execute este script do diretório RL/scripts/training/")
        return False
    
    # Criar backup do train.py original
    original_train = Path('../../train.py')
    backup_train = Path('../../train_backup.py')
    
    if not backup_train.exists():
        print("📋 Criando backup do train.py original...")
        original_train.rename(backup_train)
    
    # Criar versão de teste
    print("🔧 Criando versão de teste (10 episódios)...")
    
    with open(backup_train, 'r') as f:
        content = f.read()
    
    # Modificar para apenas 10 episódios
    content = content.replace('EPISODES = 10000', 'EPISODES = 10')
    content = content.replace('BATCH_SIZE = 64', 'BATCH_SIZE = 32')
    
    with open(original_train, 'w') as f:
        f.write(content)
    
    try:
        print("🚀 Executando teste de treinamento...")
        print("   (10 episódios apenas)")
        print("")
        
        # Mudar para o diretório RL e executar treinamento
        os.chdir('../..')
        result = subprocess.run([sys.executable, 'train.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Teste concluído com sucesso!")
            print("📊 Saída do teste:")
            print("-" * 40)
            
            # Mostrar apenas as últimas linhas relevantes
            lines = result.stdout.split('\n')
            relevant_lines = []
            for line in lines:
                if any(keyword in line for keyword in ['Training Progress', 'Resumo', 'Modelo salvo', 'Treinamento concluído']):
                    relevant_lines.append(line)
            
            for line in relevant_lines[-10:]:  # Últimas 10 linhas relevantes
                print(line)
            
            return True
        else:
            print("❌ Erro durante o teste:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Teste demorou muito (>5 minutos) - interrompido")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False
    finally:
        # Restaurar arquivo original
        print("🔄 Restaurando arquivo original...")
        if original_train.exists():
            original_train.unlink()
        backup_train.rename(original_train)

def main():
    success = test_training()
    
    if success:
        print("\n🎉 TUDO FUNCIONANDO!")
        print("Agora você pode executar o treinamento completo:")
        print("   ./scripts/training/run_training.sh")
        print("   ou")
        print("   python3 scripts/training/auto_train.py")
    else:
        print("\n❌ TESTE FALHOU")
        print("Verifique as dependências e tente novamente")

if __name__ == "__main__":
    main() 