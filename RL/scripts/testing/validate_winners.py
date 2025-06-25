#!/usr/bin/env python3
"""
Script para validar se os modelos vencedores realmente são bons ou apenas tiveram sorte
Baseado na ideia original proposta pelo usuário
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força CPU

from dqn_agent import DQNAgent
from knight_env import KnightTourEnv
import numpy as np
import pandas as pd

def load_training_data():
    """Carrega e analisa os dados de treinamento"""
    df = pd.read_csv('logs/training_log.csv')
    print(f"📊 Dados carregados: {len(df)} episódios")
    
    # Encontra os ranges com mais vitórias
    window_size = 100
    best_ranges = []
    
    for start in range(5000, len(df), window_size):  # Foca em episódios > 5000
        end = min(start + window_size, len(df))
        window_data = df.iloc[start:end]
        
        wins = window_data['Win'].sum()
        total = len(window_data)
        win_rate = wins / total if total > 0 else 0
        
        if win_rate > 0.8:  # Apenas ranges com alta taxa de vitória
            avg_score = window_data['Score'].mean()
            best_ranges.append({
                'range_start': start,
                'range_end': end - 1,
                'win_rate': win_rate,
                'wins': wins,
                'total': total,
                'avg_score': avg_score
            })
    
    # Ordena por taxa de vitória
    best_ranges.sort(key=lambda x: x['win_rate'], reverse=True)
    
    print(f"\n🏆 Melhores ranges encontrados (>80% vitórias):")
    for range_info in best_ranges[:5]:
        print(f"   Episódios {range_info['range_start']:4d}-{range_info['range_end']:4d}: "
              f"{range_info['win_rate']:.1%} vitórias "
              f"({range_info['wins']}/{range_info['total']})")
    
    return best_ranges

def get_available_models():
    """Lista modelos disponíveis"""
    models = []
    models_dir = 'models/'
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.h5'):
                try:
                    episode_num = int(file.split('_e')[1].split('.')[0])
                    models.append({
                        'filename': file,
                        'episode': episode_num,
                        'path': os.path.join(models_dir, file)
                    })
                except:
                    continue
    
    return sorted(models, key=lambda x: x['episode'])

def select_winner_models(best_ranges, available_models):
    """
    Seleciona modelos vencedores baseado nos melhores ranges
    Implementa a lógica proposta pelo usuário
    """
    print(f"\n🎯 Selecionando modelos vencedores...")
    
    selected_models = []
    
    # Estratégia: pegar modelos dos ranges com melhor performance
    for range_info in best_ranges[:3]:  # Top 3 ranges
        range_center = (range_info['range_start'] + range_info['range_end']) / 2
        
        # Encontra modelo mais próximo do centro do range
        closest_model = None
        min_distance = float('inf')
        
        for model in available_models:
            if model['episode'] >= 5000:  # Apenas modelos tardios
                distance = abs(model['episode'] - range_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_model = model
        
        if closest_model and closest_model not in selected_models:
            selected_models.append(closest_model)
    
    # Adiciona alguns modelos finais (últimos salvos)
    for model in available_models[-3:]:
        if model not in selected_models:
            selected_models.append(model)
    
    print(f"✅ Modelos selecionados para validação:")
    for model in selected_models:
        print(f"   - Episódio {model['episode']:4d}: {model['filename']}")
    
    return selected_models

def test_model_robustness(model_path, episode_num, num_tests=20):
    """
    Testa se o modelo realmente é bom ou apenas teve sorte
    Implementação baseada na ideia original do usuário
    """
    env = KnightTourEnv(board_size=5)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    # Carrega o agente
    agent = DQNAgent(state_shape, action_size)
    
    try:
        agent.load(model_path)
        agent.epsilon = 0.0  # Sem exploração - modo teste
    except Exception as e:
        print(f"❌ Erro ao carregar {model_path}: {e}")
        return None
    
    vitorias = 0
    total_episodios = num_tests
    scores = []
    game_details = []
    
    print(f"🧪 Testando modelo episódio {episode_num} ({total_episodios} testes)...")
    
    for teste in range(total_episodios):
        state = env.reset()
        done = False
        steps = 0
        max_steps = 1000  # Evita loops infinitos
        
        while not done and steps < max_steps:
            valid_moves_mask = env._get_valid_moves_mask()
            
            if not valid_moves_mask.any():
                # Travou - sem movimentos válidos
                break
            
            action = agent.act(state, valid_moves_mask)
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1
        
        # Verifica se conseguiu completar o tour
        final_score = env.visited_count
        scores.append(final_score)
        
        if env.visited_count == env.total_squares:  # 25 para board 5x5
            vitorias += 1
            game_details.append({'test': teste, 'result': 'WIN', 'score': final_score, 'steps': steps})
        else:
            game_details.append({'test': teste, 'result': 'LOSS', 'score': final_score, 'steps': steps})
    
    # Calcula estatísticas
    taxa_vitoria = vitorias / total_episodios
    score_medio = np.mean(scores)
    score_std = np.std(scores)
    
    result = {
        'episode': episode_num,
        'model_path': model_path,
        'vitorias': vitorias,
        'total_episodios': total_episodios,
        'taxa_vitoria': taxa_vitoria,
        'score_medio': score_medio,
        'score_std': score_std,
        'score_min': min(scores),
        'score_max': max(scores),
        'game_details': game_details
    }
    
    return result

def main():
    """Função principal - implementa a ideia do usuário"""
    print("🎯 VALIDAÇÃO DOS MODELOS VENCEDORES")
    print("Testando se os modelos que venceram no treinamento")
    print("realmente são bons ou apenas tiveram sorte!")
    print("=" * 60)
    
    # 1. Analisa dados de treinamento para encontrar melhores ranges
    best_ranges = load_training_data()
    
    # 2. Lista modelos disponíveis
    available_models = get_available_models()
    print(f"\n📁 Encontrados {len(available_models)} modelos disponíveis")
    
    # 3. Seleciona modelos vencedores baseado na análise
    modelos_vencedores = select_winner_models(best_ranges, available_models)
    
    # 4. Testa robustez de cada modelo
    print(f"\n🚀 INICIANDO TESTES DE VALIDAÇÃO")
    print("=" * 60)
    
    resultados = {}
    
    for i, model in enumerate(modelos_vencedores, 1):
        print(f"\n[{i}/{len(modelos_vencedores)}] ", end="")
        
        result = test_model_robustness(
            model['path'], 
            model['episode'], 
            num_tests=20  # Teste rápido mas significativo
        )
        
        if result:
            resultados[model['path']] = result
            print(f"   ✅ {result['taxa_vitoria']:.1%} vitórias "
                  f"({result['vitorias']}/{result['total_episodios']}), "
                  f"Score médio: {result['score_medio']:.1f}")
    
    # 5. Exibe resultados finais
    print(f"\n🏆 RESULTADOS FINAIS - VALIDAÇÃO DE ROBUSTEZ")
    print("=" * 80)
    
    # Ordena por taxa de vitória
    sorted_results = sorted(resultados.items(), 
                          key=lambda x: x[1]['taxa_vitoria'], 
                          reverse=True)
    
    print(f"{'Episódio':>8} {'Taxa Vitória':>12} {'Vitórias':>9} {'Score':>10} {'Avaliação':>15}")
    print("-" * 80)
    
    for model_path, resultado in sorted_results:
        episode = resultado['episode']
        taxa = resultado['taxa_vitoria']
        vitorias = resultado['vitorias']
        total = resultado['total_episodios']
        score = resultado['score_medio']
        
        # Avaliação da robustez
        if taxa >= 0.95:
            avaliacao = "🎉 EXCELENTE"
        elif taxa >= 0.8:
            avaliacao = "👍 MUITO BOM"
        elif taxa >= 0.6:
            avaliacao = "✅ BOM"
        elif taxa >= 0.4:
            avaliacao = "⚠️ REGULAR"
        else:
            avaliacao = "❌ FRACO"
        
        print(f"{episode:>8} {taxa:>11.1%} {vitorias:>4}/{total:<3} {score:>9.1f} {avaliacao:>15}")
    
    # Conclusão
    if sorted_results:
        melhor = sorted_results[0][1]
        print(f"\n🥇 MELHOR MODELO VALIDADO:")
        print(f"   Episódio: {melhor['episode']}")
        print(f"   Taxa de vitória: {melhor['taxa_vitoria']:.1%}")
        print(f"   Consistência: Score {melhor['score_min']}-{melhor['score_max']} "
              f"(σ={melhor['score_std']:.1f})")
        
        if melhor['taxa_vitoria'] >= 0.9:
            print(f"   🎯 CONCLUSÃO: Este modelo é genuinamente robusto!")
            print(f"   Não foi sorte - ele realmente aprendeu a solucionar o problema.")
        elif melhor['taxa_vitoria'] >= 0.7:
            print(f"   🎯 CONCLUSÃO: Modelo confiável, mas pode melhorar.")
        else:
            print(f"   🎯 CONCLUSÃO: Modelo pode ter tido sorte no treinamento.")
            print(f"   Recomenda-se mais treinamento ou ajuste de hiperparâmetros.")
    
    return resultados

if __name__ == "__main__":
    results = main() 