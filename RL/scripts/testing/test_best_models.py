import pandas as pd
import numpy as np
from dqn_agent import DQNAgent
from knight_env import KnightTourEnv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# Força uso da CPU para evitar problemas de GPU/CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

class BestModelsTester:
    """
    Classe para testar e validar os melhores modelos DQN treinados
    """
    
    def __init__(self, board_size=5, csv_path='logs/training_log.csv', models_dir='models/'):
        self.board_size = board_size
        self.csv_path = csv_path
        self.models_dir = models_dir
        self.env = KnightTourEnv(board_size=board_size)
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        
    def load_training_data(self):
        """Carrega os dados de treinamento"""
        print("📊 Carregando dados de treinamento...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Dados carregados: {len(self.df)} episódios")
        return self.df
    
    def analyze_training_performance(self, window_size=100):
        """Analisa a performance do treinamento por ranges/janelas"""
        print(f"\n🔍 Analisando performance por janelas de {window_size} episódios...")
        
        # Cria análise por janelas
        ranges_analysis = []
        
        for start in range(0, len(self.df), window_size):
            end = min(start + window_size, len(self.df))
            window_data = self.df.iloc[start:end]
            
            wins = window_data['Win'].sum()
            total = len(window_data)
            win_rate = wins / total if total > 0 else 0
            
            avg_score = window_data['Score'].mean()
            max_visited = window_data['Max_Visited'].max()
            avg_reward = window_data['Avg_Reward'].mean()
            
            ranges_analysis.append({
                'range_start': start,
                'range_end': end - 1,
                'episode_center': (start + end - 1) / 2,
                'total_episodes': total,
                'wins': wins,
                'win_rate': win_rate,
                'avg_score': avg_score,
                'max_visited': max_visited,
                'avg_reward': avg_reward
            })
        
        self.ranges_df = pd.DataFrame(ranges_analysis)
        
        # Identifica os melhores ranges
        self.best_ranges = self.ranges_df.nlargest(10, 'win_rate')
        
        print(f"\n🏆 Top 10 melhores ranges por taxa de vitória:")
        for _, row in self.best_ranges.iterrows():
            print(f"   Episódios {int(row['range_start']):4d}-{int(row['range_end']):4d}: "
                  f"{row['win_rate']:.1%} vitórias ({int(row['wins']):2d}/{int(row['total_episodes']):2d}), "
                  f"Score médio: {row['avg_score']:.1f}")
        
        return self.ranges_df, self.best_ranges
    
    def get_available_models(self):
        """Lista os modelos disponíveis"""
        print(f"\n📁 Verificando modelos disponíveis em {self.models_dir}...")
        
        model_files = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('.h5'):
                    # Extrai o número do episódio do nome do arquivo
                    try:
                        episode_num = int(file.split('_e')[1].split('.')[0])
                        model_files.append({
                            'filename': file,
                            'episode': episode_num,
                            'path': os.path.join(self.models_dir, file)
                        })
                    except:
                        print(f"⚠️  Formato de nome não reconhecido: {file}")
        
        self.available_models = sorted(model_files, key=lambda x: x['episode'])
        print(f"✅ Encontrados {len(self.available_models)} modelos")
        
        return self.available_models
    
    def select_models_to_test(self, strategy='best_ranges', max_models=10):
        """Seleciona os modelos mais promissores para teste"""
        print(f"\n🎯 Selecionando modelos usando estratégia: {strategy}")
        
        if strategy == 'best_ranges':
            # Seleciona modelos dos melhores ranges
            selected_episodes = set()
            
            for _, best_range in self.best_ranges.head(5).iterrows():
                range_center = int(best_range['episode_center'])
                
                # Procura modelo mais próximo do centro do range
                closest_model = None
                min_distance = float('inf')
                
                for model in self.available_models:
                    distance = abs(model['episode'] - range_center)
                    if distance < min_distance:
                        min_distance = distance
                        closest_model = model
                
                if closest_model and closest_model['episode'] not in selected_episodes:
                    selected_episodes.add(closest_model['episode'])
            
            # Adiciona modelos dos episódios finais (últimos modelos)
            for model in self.available_models[-3:]:
                selected_episodes.add(model['episode'])
            
            # Seleciona os modelos correspondentes
            self.selected_models = [
                model for model in self.available_models 
                if model['episode'] in selected_episodes
            ][:max_models]
        
        elif strategy == 'high_episodes':
            # Seleciona apenas modelos de episódios altos (>= 5000)
            self.selected_models = [
                model for model in self.available_models 
                if model['episode'] >= 5000
            ][:max_models]
        
        elif strategy == 'all_available':
            # Testa todos os modelos disponíveis
            self.selected_models = self.available_models[:max_models]
        
        self.selected_models = sorted(self.selected_models, key=lambda x: x['episode'])
        
        print(f"✅ Selecionados {len(self.selected_models)} modelos para teste:")
        for model in self.selected_models:
            print(f"   - Episódio {model['episode']:4d}: {model['filename']}")
        
        return self.selected_models
    
    def test_model_robustness(self, model_path, episode_num, num_tests=100, verbose=False):
        """Testa a robustez de um modelo específico"""
        if verbose:
            print(f"🧪 Testando modelo do episódio {episode_num} ({num_tests} testes)...")
        
        # Carrega o agente e o modelo
        agent = DQNAgent(self.state_shape, self.action_size)
        
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"❌ Erro ao carregar modelo {model_path}: {e}")
            return None
        
        # Desativa exploração para teste
        agent.epsilon = 0.0
        
        results = {
            'episode': episode_num,
            'model_path': model_path,
            'wins': 0,
            'total_tests': num_tests,
            'scores': [],
            'visited_counts': [],
            'game_lengths': [],
            'stuck_games': 0,
            'win_rate': 0.0
        }
        
        for test_num in range(num_tests):
            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 1000  # Previne loops infinitos
            
            while not done and steps < max_steps:
                valid_moves_mask = self.env._get_valid_moves_mask()
                
                if not valid_moves_mask.any():
                    # Sem movimentos válidos - jogo travado
                    results['stuck_games'] += 1
                    break
                
                action = agent.act(state, valid_moves_mask)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                steps += 1
            
            # Registra resultados do jogo
            results['scores'].append(self.env.visited_count)
            results['visited_counts'].append(self.env.visited_count)
            results['game_lengths'].append(steps)
            
            if self.env.visited_count == self.env.total_squares:
                results['wins'] += 1
        
        # Calcula estatísticas finais
        results['win_rate'] = results['wins'] / num_tests
        results['avg_score'] = np.mean(results['scores'])
        results['max_score'] = np.max(results['scores'])
        results['std_score'] = np.std(results['scores'])
        results['avg_game_length'] = np.mean(results['game_lengths'])
        
        if verbose:
            print(f"   ✅ Resultados: {results['win_rate']:.1%} vitórias "
                  f"({results['wins']}/{num_tests}), "
                  f"Score médio: {results['avg_score']:.1f}")
        
        return results
    
    def test_all_selected_models(self, num_tests=50, save_results=True):
        """Testa todos os modelos selecionados"""
        print(f"\n🚀 Iniciando testes de robustez ({num_tests} testes por modelo)...")
        
        self.test_results = []
        
        for i, model in enumerate(self.selected_models, 1):
            print(f"\n[{i}/{len(self.selected_models)}] Testando episódio {model['episode']}...")
            
            result = self.test_model_robustness(
                model['path'], 
                model['episode'], 
                num_tests=num_tests,
                verbose=True
            )
            
            if result:
                self.test_results.append(result)
        
        # Ordena resultados por taxa de vitória
        self.test_results = sorted(self.test_results, key=lambda x: x['win_rate'], reverse=True)
        
        print(f"\n🏆 RESULTADOS FINAIS - Top Modelos por Taxa de Vitória:")
        print("=" * 80)
        print(f"{'Episódio':>8} {'Taxa Vitória':>12} {'Vitórias':>9} {'Score Médio':>12} "
              f"{'Score Máx':>10} {'Jogos Travados':>14}")
        print("-" * 80)
        
        for result in self.test_results:
            print(f"{result['episode']:>8} {result['win_rate']:>11.1%} "
                  f"{result['wins']:>4}/{result['total_tests']:<3} "
                  f"{result['avg_score']:>11.1f} {result['max_score']:>10.0f} "
                  f"{result['stuck_games']:>14}")
        
        if save_results:
            self.save_test_results()
        
        return self.test_results
    
    def save_test_results(self, filename='test_results.json'):
        """Salva os resultados dos testes"""
        results_path = os.path.join(self.models_dir, filename)
        
        # Prepara dados para serialização JSON
        serializable_results = []
        for result in self.test_results:
            clean_result = result.copy()
            # Converte arrays numpy e tipos numpy para tipos Python nativos
            for key in ['scores', 'visited_counts', 'game_lengths']:
                if key in clean_result:
                    clean_result[key] = [int(x) for x in clean_result[key]]
            
            # Converte outros valores numpy para tipos Python nativos
            for key in ['wins', 'total_tests', 'stuck_games', 'episode', 'max_score']:
                if key in clean_result:
                    clean_result[key] = int(clean_result[key])
            
            for key in ['win_rate', 'avg_score', 'std_score', 'avg_game_length']:
                if key in clean_result:
                    clean_result[key] = float(clean_result[key])
            
            serializable_results.append(clean_result)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Resultados salvos em: {results_path}")
    
    def plot_analysis(self, save_plots=True):
        """Cria visualizações da análise"""
        print("\n📊 Gerando visualizações...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise dos Melhores Modelos DQN - Knight Tour', fontsize=16)
        
        # 1. Taxa de vitória por range de episódios
        axes[0, 0].plot(self.ranges_df['episode_center'], self.ranges_df['win_rate'], 'b-', alpha=0.7)
        axes[0, 0].scatter(self.best_ranges['episode_center'], self.best_ranges['win_rate'], 
                          color='red', s=50, zorder=5)
        axes[0, 0].set_title('Taxa de Vitória por Range de Episódios')
        axes[0, 0].set_xlabel('Episódio (Centro do Range)')
        axes[0, 0].set_ylabel('Taxa de Vitória')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribuição de scores nos testes
        if hasattr(self, 'test_results') and len(self.test_results) > 0:
            all_scores = []
            labels = []
            for result in self.test_results[:5]:  # Top 5 modelos
                all_scores.extend(result['scores'])
                labels.extend([f"Ep {result['episode']}"] * len(result['scores']))
            
            if all_scores:
                scores_df = pd.DataFrame({'Score': all_scores, 'Modelo': labels})
                sns.boxplot(data=scores_df, x='Modelo', y='Score', ax=axes[0, 1])
                axes[0, 1].set_title('Distribuição de Scores - Top 5 Modelos')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Taxa de vitória dos modelos testados
        if hasattr(self, 'test_results') and len(self.test_results) > 0:
            episodes = [r['episode'] for r in self.test_results]
            win_rates = [r['win_rate'] for r in self.test_results]
            
            bars = axes[1, 0].bar(range(len(episodes)), win_rates, 
                                 color='green', alpha=0.7)
            axes[1, 0].set_title('Taxa de Vitória por Modelo Testado')
            axes[1, 0].set_xlabel('Modelos (ordenados por performance)')
            axes[1, 0].set_ylabel('Taxa de Vitória')
            axes[1, 0].set_xticks(range(len(episodes)))
            axes[1, 0].set_xticklabels([f"Ep {ep}" for ep in episodes], rotation=45)
            
            # Adiciona valores nas barras
            for bar, rate in zip(bars, win_rates):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{rate:.1%}', ha='center', va='bottom', fontsize=8)
        
        # 4. Correlação entre score médio e taxa de vitória
        if hasattr(self, 'test_results') and len(self.test_results) > 0:
            avg_scores = [r['avg_score'] for r in self.test_results]
            win_rates = [r['win_rate'] for r in self.test_results]
            episodes = [r['episode'] for r in self.test_results]
            
            scatter = axes[1, 1].scatter(avg_scores, win_rates, c=episodes, 
                                       cmap='viridis', s=60, alpha=0.7)
            axes[1, 1].set_title('Score Médio vs Taxa de Vitória')
            axes[1, 1].set_xlabel('Score Médio')
            axes[1, 1].set_ylabel('Taxa de Vitória')
            plt.colorbar(scatter, ax=axes[1, 1], label='Episódio')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.models_dir, 'model_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráficos salvos em: {plot_path}")
        
        plt.show()
    
    def run_complete_analysis(self, num_tests=50, strategy='best_ranges'):
        """Executa a análise completa"""
        print("🎯 INICIANDO ANÁLISE COMPLETA DOS MELHORES MODELOS")
        print("=" * 60)
        
        # 1. Carrega dados de treinamento
        self.load_training_data()
        
        # 2. Analisa performance por ranges
        self.analyze_training_performance()
        
        # 3. Identifica modelos disponíveis
        self.get_available_models()
        
        # 4. Seleciona melhores modelos
        self.select_models_to_test(strategy=strategy)
        
        # 5. Testa robustez dos modelos
        self.test_all_selected_models(num_tests=num_tests)
        
        # 6. Cria visualizações
        self.plot_analysis()
        
        print(f"\n🎉 ANÁLISE COMPLETA FINALIZADA!")
        print(f"✅ Melhor modelo: Episódio {self.test_results[0]['episode']} "
              f"({self.test_results[0]['win_rate']:.1%} vitórias)")
        
        return self.test_results

# Função principal para uso direto
def main():
    # Configurações
    BOARD_SIZE = 5
    NUM_TESTS = 50  # Número de testes por modelo
    STRATEGY = 'best_ranges'  # 'best_ranges', 'high_episodes', 'all_available'
    
    # Cria o testador
    tester = BestModelsTester(board_size=BOARD_SIZE)
    
    # Executa análise completa
    results = tester.run_complete_analysis(num_tests=NUM_TESTS, strategy=STRATEGY)
    
    return results

if __name__ == "__main__":
    results = main() 