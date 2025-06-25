#!/usr/bin/env python3
"""
Script para iniciar o jogo Cavalo Solitário
Inicia tanto o servidor web quanto o servidor Flask para o AI
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    try:
        import flask
        import flask_cors
        print("✓ Dependências Flask encontradas")
    except ImportError as e:
        print(f"✗ Dependência faltando: {e}")
        print("Execute: pip install flask flask-cors")
        return False
    return True

def start_flask_server():
    """Inicia o servidor Flask para o AI"""
    print("🚀 Iniciando servidor Flask para AI...")
    try:
        # Muda para o diretório RL
        os.chdir('RL')
        
        # Inicia o servidor Flask
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Aguarda um pouco para verificar se iniciou corretamente
        time.sleep(3)
        
        if process.poll() is None:
            print("✓ Servidor Flask iniciado na porta 5001")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"✗ Erro ao iniciar servidor Flask: {stderr}")
            return None
            
    except Exception as e:
        print(f"✗ Erro ao iniciar servidor Flask: {e}")
        return None

def start_web_server():
    """Inicia o servidor web para o jogo"""
    print("🌐 Iniciando servidor web...")
    try:
        # Volta para o diretório raiz
        os.chdir('..')
        
        # Inicia o servidor HTTP simples
        process = subprocess.Popen([
            sys.executable, '-m', 'http.server', '8000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Aguarda um pouco para verificar se iniciou corretamente
        time.sleep(2)
        
        if process.poll() is None:
            print("✓ Servidor web iniciado na porta 8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"✗ Erro ao iniciar servidor web: {stderr}")
            return None
            
    except Exception as e:
        print(f"✗ Erro ao iniciar servidor web: {e}")
        return None

def main():
    """Função principal"""
    print("🎮 Iniciando Jogo Cavalo Solitário")
    print("=" * 50)
    
    # Verifica dependências
    if not check_dependencies():
        return
    
    # Verifica se existe um modelo treinado
    model_files = list(Path('models').glob('*.h5'))
    if not model_files:
        print("⚠️  Nenhum modelo treinado encontrado em models/")
        print("   O recurso de AI Hint pode não funcionar.")
        print("   Execute o treinamento primeiro: python RL/train.py")
    else:
        print(f"✓ Modelo encontrado: {model_files[-1].name}")
    
    # Inicia os servidores
    flask_process = start_flask_server()
    web_process = start_web_server()
    
    if not flask_process or not web_process:
        print("✗ Falha ao iniciar um dos servidores")
        return
    
    print("\n🎯 Jogo iniciado com sucesso!")
    print("📱 Acesse: http://localhost:8000")
    print("🤖 AI Hint disponível via Flask na porta 5001")
    print("\n⏹️  Pressione Ctrl+C para parar os servidores")
    
    try:
        # Mantém os processos rodando
        while True:
            time.sleep(1)
            
            # Verifica se algum processo morreu
            if flask_process.poll() is not None:
                print("✗ Servidor Flask parou inesperadamente")
                break
                
            if web_process.poll() is not None:
                print("✗ Servidor web parou inesperadamente")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Parando servidores...")
        
        # Para os processos
        if flask_process:
            flask_process.terminate()
            flask_process.wait()
            
        if web_process:
            web_process.terminate()
            web_process.wait()
            
        print("✓ Servidores parados")

if __name__ == "__main__":
    main() 