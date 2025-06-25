#!/usr/bin/env python3
"""
Script rápido para testar os melhores modelos DQN do Knight Tour
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força CPU

import numpy as np
from dqn_agent import DQNAgent
from knight_env import KnightTourEnv
import pandas as pd

def quick_test_model(model_path, episode_num, num_tests=20):
    """Testa rapidamente um modelo específico"""
    env = KnightTourEnv(board_size=5)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    
    try:
        agent.load(model_path)
        agent.epsilon = 0.0  # Sem exploração
    except Exception as e:
        print(f"❌ Erro ao carregar {model_path}: {e}")
        return None
    
    wins = 0
    scores = []
    
    for i in range(num_tests):
        state = env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            valid_moves = env._get_valid_moves_mask()
            if not valid_moves.any():
                break
            
            action = agent.act(state, valid_moves)
            state, reward, done, info = env.step(action)
            steps += 1
        
        scores.append(env.visited_count)
        if env.visited_count == 25:  # 5x5 board
            wins += 1
    
    win_rate = wins / num_tests
    avg_score = np.mean(scores)
    
    return {
        'episode': episode_num,
        'model_path': model_path,
        'win_rate': win_rate,
        'wins': wins,
        'total_tests': num_tests,
        'avg_score': avg_score,
        'max_score': max(scores),
        'min_score': min(scores)
    }

def main():
    print("🎯 TESTE RÁPIDO DOS MELHORES MODELOS")
    print("=" * 50)
    
    # Modelos dos melhores ranges identificados
    best_models = [
        'models/knight_tour_dqn_b5_e5200.h5',  # Melhor range: 98% vitórias
        'models/knight_tour_dqn_b5_e5900.h5',  # 97% vitórias
        'models/knight_tour_dqn_b5_e6400.h5',  # 97% vitórias
        'models/knight_tour_dqn_b5_e5700.h5',  # 95% vitórias
        'models/knight_tour_dqn_b5_e6500.h5',  # Último modelo
    ]
    
    results = []
    
    for i, model_path in enumerate(best_models, 1):
        if not os.path.exists(model_path):
            print(f"⚠️  Modelo não encontrado: {model_path}")
            continue
        
        episode_num = int(model_path.split('_e')[1].split('.')[0])
        print(f"\n[{i}/{len(best_models)}] Testando episódio {episode_num}...")
        
        result = quick_test_model(model_path, episode_num, num_tests=30)
        if result:
            results.append(result)
            print(f"   ✅ {result['win_rate']:.1%} vitórias ({result['wins']}/{result['total_tests']}), "
                  f"Score médio: {result['avg_score']:.1f}")
    
    # Ordena por taxa de vitória
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    print(f"\n🏆 RESULTADOS FINAIS:")
    print("-" * 50)
    print(f"{'Episódio':>8} {'Taxa':>6} {'Vitórias':>9} {'Score Médio':>12}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['episode']:>8} {result['win_rate']:>5.1%} "
              f"{result['wins']:>4}/{result['total_tests']:<3} "
              f"{result['avg_score']:>11.1f}")
    
    if results:
        best = results[0]
        print(f"\n🥇 MELHOR MODELO: Episódio {best['episode']}")
        print(f"   Taxa de vitória: {best['win_rate']:.1%}")
        print(f"   Score médio: {best['avg_score']:.1f}")
        print(f"   Range de scores: {best['min_score']}-{best['max_score']}")
        
        if best['win_rate'] >= 0.9:
            print(f"   🎉 EXCELENTE! Este modelo é muito robusto!")
        elif best['win_rate'] >= 0.7:
            print(f"   👍 BOM! Este modelo é confiável!")
        else:
            print(f"   ⚠️  Este modelo pode ter sido sortudo no treinamento.")
    
    return results

if __name__ == "__main__":
    results = main() 