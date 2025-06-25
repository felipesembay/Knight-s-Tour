import gym
import numpy as np
from dqn_agent import DQNAgent
from knight_env import KnightTourEnv
from tqdm import tqdm
import csv
import os
import tensorflow as tf

# --- Verificação da GPU ---
print("=== Configuração da GPU ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs disponíveis: {len(tf.config.list_physical_devices('GPU'))}")
for gpu in tf.config.list_physical_devices('GPU'):
    print(f"  - {gpu}")
print("==========================\n")

# --- Configurações ---
BOARD_SIZE = 6
EPISODES = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10 # Atualizar a target network a cada 10 episódios
MAX_STEPS_PER_EPISODE = 1000 # Aumentado

# --- Criar pastas se não existirem ---
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Verificar se as pastas foram criadas
print(f"Pasta 'models' existe: {os.path.exists('models')}")
print(f"Pasta 'logs' existe: {os.path.exists('logs')}")

def main():
    # --- Inicialização ---
    env = KnightTourEnv(board_size=BOARD_SIZE)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQNAgent(
        state_shape=state_shape, 
        action_size=action_size,
        epsilon_decay=0.995 # Decaimento mais rápido
    )

    # --- Logging ---
    log_file = 'logs/training_log.csv'
    log_header = ['Episode', 'Score', 'Max_Visited', 'Avg_Reward', 'Invalid_Moves', 'Epsilon', 'Win']
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)

    # Lista para armazenar os últimos 100 episódios
    last_100 = []

    # --- Loop de Treinamento ---
    for e in tqdm(range(EPISODES), desc="Training Progress"):
        state = env.reset()
        total_reward = 0
        invalid_move_count = 0
        
        for time in range(MAX_STEPS_PER_EPISODE):
            valid_moves_mask = env._get_valid_moves_mask()
            action = agent.act(state, valid_moves_mask)
            
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            if info.get('status') == 'invalid_move':
                invalid_move_count += 1
                
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break

        # Treinamento do agente (replay)
        agent.replay(BATCH_SIZE)

        # Atualiza a target network periodicamente
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        # Logging
        score = time + 1
        max_visited = env.visited_count
        avg_reward = total_reward / score if score > 0 else 0
        win = 1 if max_visited == env.total_squares else 0  # Vitória se visitou todas as casas
        
        # Debug: verificar se o arquivo de log existe
        if e == 0:
            print(f"Arquivo de log: {log_file}")
            print(f"Arquivo existe: {os.path.exists(log_file)}")
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e, score, max_visited, f"{avg_reward:.2f}", invalid_move_count, f"{agent.epsilon:.4f}", win])
        
        # Debug: verificar se o arquivo foi escrito (apenas nos primeiros episódios)
        if e < 5:
            print(f"Episódio {e}: Log escrito - Score: {score}, Max Visited: {max_visited}, Win: {win}")

        # Guardar desempenho atual
        last_100.append({
            'episode': e,
            'score': score,
            'visited': max_visited,
            'win': win,
            'epsilon': agent.epsilon,
            'avg_reward': avg_reward,
            'invalid_moves': invalid_move_count
        })

        # Manter apenas os últimos 100 episódios
        if len(last_100) > 100:
            last_100.pop(0)

        # Reiniciar epsilon a cada 1000 episódios se não houve vitória
        if (e + 1) % 1000 == 0:
            if win == 0:
                print(f"\nNenhuma vitória nos últimos 1000 episódios. Reiniciando epsilon para 0.1.")
                agent.epsilon = max(agent.epsilon, 0.2)
                print(f"Episode {e+1}: Epsilon {agent.epsilon:.4f} | Max Visited {max_visited} | Invalid Moves {invalid_move_count}")

        # Ao final de cada grupo de 100 episódios
        if (e + 1) % 100 == 0:
            # Encontra o melhor episódio (prioriza vitórias, depois maior número de casas visitadas)
            best_episode = max(last_100, key=lambda x: (x['win'], x['visited'], -x['episode']))
            
            # Conta quantas vitórias houve no grupo
            wins_in_group = sum(1 for ep in last_100 if ep['win'] == 1)
            
            print(f"\n[Resumo dos Episódios {e - 99} a {e+1}]")
            print(f"Melhor Episódio: {best_episode['episode']} | Score: {best_episode['score']} | Max Visited: {best_episode['visited']}/{env.total_squares} | Win: {best_episode['win']} | Epsilon: {best_episode['epsilon']:.4f}")
            print(f"Vitórias no grupo: {wins_in_group}/100 | Média de casas visitadas: {sum(ep['visited'] for ep in last_100)/len(last_100):.1f}")

            # Salva modelo
            model_path = f"models/6x6/knight_tour_dqn_b{BOARD_SIZE}_e{e+1}.h5"
            agent.save(model_path)
            print(f"Modelo salvo em: {model_path}")
            print(f"Arquivo do modelo existe: {os.path.exists(model_path)}")

    print("Treinamento concluído.")
    final_model_path = f"models/6x6/knight_tour_dqn_b{BOARD_SIZE}_final.h5"
    agent.save(final_model_path)

if __name__ == "__main__":
    main() 