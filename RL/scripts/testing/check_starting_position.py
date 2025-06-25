#!/usr/bin/env python3
"""
Script para verificar qual posição inicial o modelo DQN usa e testá-la
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força CPU

from dqn_agent import DQNAgent
from knight_env import KnightTourEnv
import numpy as np

def test_model_starting_position():
    """Testa qual posição inicial o modelo usa"""
    
    print("🔍 VERIFICANDO POSIÇÃO INICIAL DO MODELO")
    print("=" * 50)
    
    # Carrega o modelo configurado
    env = KnightTourEnv(board_size=5)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    
    model_path = 'models/knight_tour_dqn_b5_e5200.h5'
    try:
        agent.load(model_path)
        agent.epsilon = 0.0
        print(f"✅ Modelo carregado: {model_path}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    print(f"\n📍 POSIÇÃO INICIAL NO AMBIENTE:")
    
    # Testa ambiente padrão
    state = env.reset()
    print(f"   Posição: {env.current_pos}")
    print(f"   Coordenadas: Linha {env.current_pos[0]}, Coluna {env.current_pos[1]}")
    print(f"   No tabuleiro 5x5: Centro ({env.board_size//2}, {env.board_size//2})")
    
    # Mostra o tabuleiro inicial
    print(f"\n🏁 TABULEIRO INICIAL:")
    render_board = env.board.copy().astype(str)
    render_board[render_board == '0'] = '.'
    render_board[render_board == '2'] = 'K'  # K = Knight (Cavalo)
    
    print("   ", end="")
    for c in range(5):
        print(f" {c}", end="")
    print()
    
    for r in range(5):
        print(f"{r}: ", end="")
        for c in range(5):
            print(f" {render_board[r][c]}", end="")
        print()
    
    print(f"\n🎯 TESTE DE VITÓRIA:")
    
    # Testa se o modelo consegue vencer a partir da posição inicial padrão
    num_tests = 5
    wins = 0
    
    for test in range(num_tests):
        state = env.reset()
        done = False
        moves = []
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            valid_moves_mask = env._get_valid_moves_mask()
            if not valid_moves_mask.any():
                break
            
            action = agent.act(state, valid_moves_mask)
            move = env.knight_moves[action]
            moves.append({
                'step': steps + 1,
                'from': env.current_pos,
                'move': move,
                'to': (env.current_pos[0] + move[0], env.current_pos[1] + move[1])
            })
            
            state, reward, done, info = env.step(action)
            steps += 1
        
        if env.visited_count == 25:
            wins += 1
            print(f"   Teste {test + 1}: ✅ VITÓRIA em {steps} movimentos")
        else:
            print(f"   Teste {test + 1}: ❌ Falhou - visitou {env.visited_count}/25")
    
    print(f"\n📊 RESULTADOS:")
    print(f"   Taxa de vitória: {wins}/{num_tests} ({wins/num_tests*100:.1f}%)")
    
    if wins > 0:
        print(f"\n✅ O modelo FUNCIONA na posição inicial padrão!")
        print(f"   Posição inicial: ({env.current_pos[0]}, {env.current_pos[1]})")
        print(f"   Para usar no jogo: comece na posição CENTRO do tabuleiro 5x5")
    else:
        print(f"\n⚠️  O modelo pode ter problema na posição inicial atual")
        print(f"   Vamos testar outras posições...")
        test_different_starting_positions(agent)

def test_different_starting_positions(agent):
    """Testa o modelo em diferentes posições iniciais"""
    print(f"\n🔄 TESTANDO DIFERENTES POSIÇÕES INICIAIS:")
    
    positions_to_test = [
        (0, 0),  # Canto superior esquerdo
        (0, 4),  # Canto superior direito
        (4, 0),  # Canto inferior esquerdo
        (4, 4),  # Canto inferior direito
        (2, 2),  # Centro (padrão)
        (1, 1),  # Próximo ao canto
        (3, 3),  # Próximo ao canto oposto
    ]
    
    results = []
    
    for pos in positions_to_test:
        env = KnightTourEnv(board_size=5)
        
        # Força a posição inicial
        env.reset()
        env.current_pos = pos
        env.board = np.zeros((5, 5), dtype=np.int8)
        env.board[pos] = 2
        env.visited_count = 1
        env.path = [pos]
        
        # Testa 3 vezes nesta posição
        wins = 0
        for _ in range(3):
            env.current_pos = pos
            env.board = np.zeros((5, 5), dtype=np.int8)
            env.board[pos] = 2
            env.visited_count = 1
            env.path = [pos]
            
            state = env._get_observation()
            done = False
            steps = 0
            
            while not done and steps < 100:
                valid_moves_mask = env._get_valid_moves_mask()
                if not valid_moves_mask.any():
                    break
                
                action = agent.act(state, valid_moves_mask)
                state, reward, done, info = env.step(action)
                steps += 1
            
            if env.visited_count == 25:
                wins += 1
        
        win_rate = wins / 3
        results.append((pos, win_rate, wins))
        print(f"   Posição {pos}: {wins}/3 vitórias ({win_rate*100:.1f}%)")
    
    # Encontra a melhor posição
    best_pos, best_rate, best_wins = max(results, key=lambda x: x[1])
    
    print(f"\n🏆 MELHOR POSIÇÃO INICIAL:")
    print(f"   Posição: {best_pos}")
    print(f"   Taxa de vitória: {best_rate*100:.1f}%")
    print(f"   Coordenadas no jogo: Linha {best_pos[0]}, Coluna {best_pos[1]}")
    
    if best_rate > 0:
        print(f"\n💡 DICA PARA O JOGO:")
        print(f"   1. Selecione tabuleiro 5x5")
        print(f"   2. Clique na posição: Linha {best_pos[0]}, Coluna {best_pos[1]}")
        print(f"   3. Use 'Dica da IA' para obter as melhores jogadas")
        print(f"   4. Siga as sugestões da IA para ganhar!")

def main():
    test_model_starting_position()

if __name__ == "__main__":
    main() 