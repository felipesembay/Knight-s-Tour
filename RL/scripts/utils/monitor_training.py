#!/usr/bin/env python3
"""
Monitor de Progresso de Treinamento
Monitora logs de treinamento em tempo real e mostra estatísticas
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse

def monitor_training_progress(board_size, log_file=None, refresh_interval=10):
    """
    Monitora o progresso do treinamento em tempo real
    
    Args:
        board_size: Tamanho do tabuleiro sendo treinado
        log_file: Arquivo de log específico (opcional)
        refresh_interval: Intervalo de atualização em segundos
    """
    
    print(f"📊 MONITOR DE TREINAMENTO - {board_size}x{board_size}")
    print("=" * 60)
    
    if not log_file:
        # Procura o log mais recente
        logs_dir = f"logs/{board_size}x{board_size}"
        if not os.path.exists(logs_dir):
            print(f"❌ Diretório de logs não encontrado: {logs_dir}")
            return
        
        log_files = [f for f in os.listdir(logs_dir) if f.startswith('training_log_') and f.endswith('.csv')]
        if not log_files:
            print(f"❌ Nenhum log de treinamento encontrado em {logs_dir}")
            return
        
        # Pega o mais recente
        log_files.sort()
        log_file = os.path.join(logs_dir, log_files[-1])
    
    print(f"📁 Monitorando: {log_file}")
    print(f"🔄 Atualizando a cada {refresh_interval} segundos")
    print("   Pressione Ctrl+C para parar")
    print("-" * 60)
    
    last_episode = 0
    
    try:
        while True:
            try:
                # Lê o arquivo de log
                df = pd.read_csv(log_file)
                
                if len(df) > last_episode:
                    # Novos episódios desde a última verificação
                    new_data = df.iloc[last_episode:]
                    
                    # Estatísticas atuais
                    current_episode = df.iloc[-1]['episode']
                    current_reward = df.iloc[-1]['reward']
                    current_visited = df.iloc[-1]['visited_count']
                    current_win = df.iloc[-1]['win']
                    current_epsilon = df.iloc[-1]['epsilon']
                    
                    # Melhores resultados até agora
                    best_reward = df['reward'].max()
                    best_visited = df['visited_count'].max()
                    total_wins = df['win'].sum()
                    
                    # Taxa de vitória nos últimos 100
                    recent_data = df.tail(100)
                    recent_win_rate = recent_data['win'].mean()
                    
                    # Taxa de progresso (casas visitadas)
                    total_squares = board_size * board_size
                    progress_rate = (best_visited / total_squares) * 100
                    
                    print(f"\r🎯 Ep {current_episode:5d} | "
                          f"Atual: R={current_reward:6.1f} V={current_visited:2d} | "
                          f"Melhor: R={best_reward:6.1f} V={best_visited:2d} | "
                          f"Progresso: {progress_rate:5.1f}% | "
                          f"Wins: {total_wins} | "
                          f"Taxa: {recent_win_rate:.1%} | "
                          f"ε: {current_epsilon:.3f}", end="")
                    
                    last_episode = len(df)
                
                time.sleep(refresh_interval)
                
            except FileNotFoundError:
                print(f"\r⏳ Aguardando arquivo de log...", end="")
                time.sleep(refresh_interval)
            except pd.errors.EmptyDataError:
                print(f"\r⏳ Aguardando dados...", end="")
                time.sleep(refresh_interval)
                
    except KeyboardInterrupt:
        print(f"\n\n📊 RESUMO FINAL:")
        
        try:
            df = pd.read_csv(log_file)
            
            print(f"   Total de episódios: {len(df)}")
            print(f"   Melhor recompensa: {df['reward'].max():.1f}")
            print(f"   Máximo visitado: {df['visited_count'].max()}/{board_size*board_size}")
            print(f"   Total de vitórias: {df['win'].sum()}")
            print(f"   Taxa de vitória geral: {df['win'].mean():.1%}")
            
            if len(df) >= 100:
                recent_win_rate = df.tail(100)['win'].mean()
                print(f"   Taxa últimos 100: {recent_win_rate:.1%}")
            
            print(f"   Epsilon final: {df.iloc[-1]['epsilon']:.3f}")
            
        except Exception as e:
            print(f"   Erro ao gerar resumo: {e}")

def plot_training_progress(board_size, log_file=None):
    """Gera gráfico do progresso de treinamento"""
    
    if not log_file:
        logs_dir = f"logs/{board_size}x{board_size}"
        log_files = [f for f in os.listdir(logs_dir) if f.startswith('training_log_') and f.endswith('.csv')]
        if not log_files:
            print(f"❌ Nenhum log encontrado")
            return
        log_files.sort()
        log_file = os.path.join(logs_dir, log_files[-1])
    
    df = pd.read_csv(log_file)
    
    # Cria gráfico com subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Recompensa ao longo do tempo
    ax1.plot(df['episode'], df['reward'])
    ax1.set_title('Recompensa por Episódio')
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Recompensa')
    ax1.grid(True)
    
    # Casas visitadas
    ax2.plot(df['episode'], df['visited_count'])
    ax2.axhline(y=board_size*board_size, color='r', linestyle='--', label='Objetivo')
    ax2.set_title('Casas Visitadas por Episódio')
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel('Casas Visitadas')
    ax2.legend()
    ax2.grid(True)
    
    # Taxa de vitória (média móvel)
    window_size = 100
    if len(df) >= window_size:
        rolling_wins = df['win'].rolling(window=window_size).mean()
        ax3.plot(df['episode'], rolling_wins)
        ax3.set_title(f'Taxa de Vitória (Média Móvel {window_size})')
        ax3.set_xlabel('Episódio')
        ax3.set_ylabel('Taxa de Vitória')
        ax3.grid(True)
    
    # Epsilon (exploração)
    ax4.plot(df['episode'], df['epsilon'])
    ax4.set_title('Epsilon (Taxa de Exploração)')
    ax4.set_xlabel('Episódio')
    ax4.set_ylabel('Epsilon')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Salva gráfico
    plot_file = f"training_progress_{board_size}x{board_size}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {plot_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Monitor de Treinamento Knight\'s Tour')
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Comando monitor
    monitor_parser = subparsers.add_parser('monitor', help='Monitora treinamento em tempo real')
    monitor_parser.add_argument('size', type=int, help='Tamanho do tabuleiro')
    monitor_parser.add_argument('--log', help='Arquivo de log específico')
    monitor_parser.add_argument('--interval', type=int, default=10, help='Intervalo de atualização (s)')
    
    # Comando plot
    plot_parser = subparsers.add_parser('plot', help='Gera gráfico do progresso')
    plot_parser.add_argument('size', type=int, help='Tamanho do tabuleiro')
    plot_parser.add_argument('--log', help='Arquivo de log específico')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'monitor':
        monitor_training_progress(args.size, args.log, args.interval)
    elif args.command == 'plot':
        plot_training_progress(args.size, args.log)

if __name__ == "__main__":
    main() 