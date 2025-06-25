#!/usr/bin/env python3
"""
Gerenciador de Modelos para Knight's Tour
Script para gerenciar, testar e configurar modelos por tamanho de tabuleiro
"""

import argparse
import os
from model_config import model_config, update_best_model
from datetime import datetime

def list_models(board_size=None):
    """Lista modelos disponíveis"""
    print("🏆 MODELOS DISPONÍVEIS POR TAMANHO")
    print("=" * 60)
    
    sizes = [board_size] if board_size else [5, 6, 7, 8]
    
    for size in sizes:
        info = model_config.get_model_info(size)
        size_key = f"{size}x{size}"
        
        print(f"\n📋 {size_key}:")
        
        if info['available']:
            status = "✅ DISPONÍVEL"
            if info.get('is_fallback'):
                status += " (FALLBACK)"
            
            print(f"   Status: {status}")
            print(f"   Arquivo: {os.path.basename(info['model_path'])}")
            print(f"   Descrição: {info['description']}")
            
            if info.get('win_rate'):
                print(f"   Taxa de vitória: {info['win_rate']}%")
            if info.get('starting_position'):
                print(f"   Posição inicial: {info['starting_position']}")
            if info.get('training_range'):
                print(f"   Origem: {info['training_range']}")
        else:
            print(f"   Status: ❌ INDISPONÍVEL")
            print(f"   Descrição: {info['description']}")
            print(f"   Ação: Treinar modelo para {size_key}")

def test_model(board_size, model_file=None, num_tests=10):
    """Testa um modelo específico"""
    import numpy as np
    from dqn_agent import DQNAgent
    from knight_env import KnightTourEnv
    
    print(f"🧪 TESTANDO MODELO - TABULEIRO {board_size}x{board_size}")
    print("=" * 50)
    
    # Determina qual modelo testar
    if model_file:
        model_path = f"models/{board_size}x{board_size}/{model_file}"
        if not os.path.exists(model_path):
            print(f"❌ Modelo não encontrado: {model_path}")
            return
    else:
        model_path = model_config.get_model_path(board_size)
        if not model_path:
            print(f"❌ Nenhum modelo disponível para {board_size}x{board_size}")
            return
    
    print(f"📁 Modelo: {os.path.basename(model_path)}")
    print(f"🎯 Testes: {num_tests}")
    
    # Configura ambiente e agente
    env = KnightTourEnv(board_size=board_size)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, epsilon=0.0)
    
    try:
        # Força CPU
        import os as os_env
        os_env.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        agent.load(model_path)
        print(f"✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Executa testes
    wins = 0
    total_moves = []
    
    print(f"\n🚀 Executando testes...")
    
    for test in range(num_tests):
        state = env.reset()
        done = False
        moves = 0
        max_moves = board_size * board_size * 3
        
        while not done and moves < max_moves:
            valid_moves_mask = env._get_valid_moves_mask()
            if not valid_moves_mask.any():
                break
            
            action = agent.act(state, valid_moves_mask)
            state, reward, done, info = env.step(action)
            moves += 1
        
        win = (env.visited_count == board_size * board_size)
        if win:
            wins += 1
            total_moves.append(moves)
        
        print(f"   Teste {test + 1:2d}: {'✅ WIN' if win else '❌ LOSE'} "
              f"({env.visited_count}/{board_size * board_size} casas, {moves} movimentos)")
    
    # Resultados
    win_rate = wins / num_tests
    avg_moves = sum(total_moves) / len(total_moves) if total_moves else 0
    
    print(f"\n📊 RESULTADOS:")
    print(f"   Taxa de vitória: {wins}/{num_tests} ({win_rate:.1%})")
    if total_moves:
        print(f"   Movimentos médios: {avg_moves:.1f}")
        print(f"   Melhor: {min(total_moves)} movimentos")
        print(f"   Pior: {max(total_moves)} movimentos")
    
    # Classifica o modelo
    if win_rate >= 0.95:
        rating = "🥇 EXCELENTE"
    elif win_rate >= 0.80:
        rating = "🥈 MUITO BOM"
    elif win_rate >= 0.60:
        rating = "🥉 BOM"
    else:
        rating = "⚠️  PRECISA MELHORAR"
    
    print(f"   Classificação: {rating}")
    
    return {
        'model_path': model_path,
        'win_rate': win_rate,
        'avg_moves': avg_moves,
        'total_tests': num_tests,
        'wins': wins
    }

def set_best_model(board_size, model_file, win_rate=None, avg_moves=None):
    """Define um modelo como melhor para um tamanho"""
    
    # Verifica se o modelo existe
    model_path = f"models/{board_size}x{board_size}/{model_file}"
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    # Se não foram fornecidos win_rate e avg_moves, executa teste
    if win_rate is None or avg_moves is None:
        print("🧪 Testando modelo para obter estatísticas...")
        results = test_model(board_size, model_file, num_tests=20)
        if not results:
            return
        
        win_rate = results['win_rate'] * 100
        avg_moves = int(results['avg_moves'])
    
    # Atualiza configuração
    validation_text = f"Validated with {win_rate}% win rate"
    starting_pos = (board_size // 2, board_size // 2)  # Centro
    
    update_best_model(
        board_size=board_size,
        model_file=model_file,
        win_rate=win_rate,
        validation=validation_text,
        avg_moves=avg_moves,
        starting_position=starting_pos,
        training_range=f"Manual validation - {datetime.now().strftime('%Y-%m-%d')}"
    )
    
    print(f"\n🎉 MODELO CONFIGURADO COMO MELHOR!")
    print(f"   Tamanho: {board_size}x{board_size}")
    print(f"   Arquivo: {model_file}")
    print(f"   Taxa de vitória: {win_rate}%")

def train_new_model(board_size, episodes=10000):
    """Inicia treinamento de novo modelo"""
    print(f"🚀 INICIANDO TREINAMENTO - {board_size}x{board_size}")
    
    # Importa e executa treinamento
    from train import train_dqn
    
    train_dqn(
        board_size=board_size,
        episodes=episodes,
        save_interval=100,
        target_win_rate=0.95,
        batch_size=32
    )

def main():
    parser = argparse.ArgumentParser(description='Gerenciador de Modelos Knight\'s Tour')
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Comando list
    list_parser = subparsers.add_parser('list', help='Lista modelos disponíveis')
    list_parser.add_argument('--size', type=int, help='Tamanho específico (5, 6, 7, 8)')
    
    # Comando test
    test_parser = subparsers.add_parser('test', help='Testa um modelo')
    test_parser.add_argument('size', type=int, help='Tamanho do tabuleiro')
    test_parser.add_argument('--model', help='Arquivo do modelo (opcional)')
    test_parser.add_argument('--tests', type=int, default=10, help='Número de testes')
    
    # Comando set-best
    set_parser = subparsers.add_parser('set-best', help='Define melhor modelo')
    set_parser.add_argument('size', type=int, help='Tamanho do tabuleiro')
    set_parser.add_argument('model', help='Arquivo do modelo')
    set_parser.add_argument('--win-rate', type=float, help='Taxa de vitória (%)')
    set_parser.add_argument('--avg-moves', type=int, help='Movimentos médios')
    
    # Comando train
    train_parser = subparsers.add_parser('train', help='Treina novo modelo')
    train_parser.add_argument('size', type=int, help='Tamanho do tabuleiro')
    train_parser.add_argument('--episodes', type=int, default=10000, help='Número de episódios')
    
    # Comando status
    status_parser = subparsers.add_parser('status', help='Mostra status geral')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_models(args.size)
    
    elif args.command == 'test':
        test_model(args.size, args.model, args.tests)
    
    elif args.command == 'set-best':
        set_best_model(args.size, args.model, args.win_rate, args.avg_moves)
    
    elif args.command == 'train':
        train_new_model(args.size, args.episodes)
    
    elif args.command == 'status':
        model_config.print_status()

if __name__ == "__main__":
    main() 