#%%
"""Treino DQN do Knight's Tour.

Uso:
    python train.py                                  # padrões abaixo, GPU se disponível
    python train.py --board-size 6 --episodes 5000 --device gpu
    python train.py --device cpu --batch-size 64

Veja também ../start_training.sh e docs/HANDOFF.md.
"""

import argparse
import csv
import os
import sys
import time

# ---------------------------------------------------------------------------
# Configurações padrão (podem ser sobrescritas pela linha de comando)
# ---------------------------------------------------------------------------
BOARD_SIZE = 5
RANDOM_START = False         # Início fixo no centro: validação rápida do reward
EPISODES = 3000              # Ciclo curto: rede menor + exploração guiada aprendem mais rápido
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 5       # Atualizar target network a cada N episódios
REPLAY_FREQ = 4              # Fazer replay a cada N passos dentro do episódio


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Treino DQN Knight's Tour")
    p.add_argument('--board-size', type=int, default=BOARD_SIZE)
    p.add_argument('--episodes', type=int, default=EPISODES)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--target-update', type=int, default=TARGET_UPDATE_FREQ, help='episódios entre updates da target')
    p.add_argument('--replay-freq', type=int, default=REPLAY_FREQ, help='passos de ambiente entre updates')
    p.add_argument('--random-start', action='store_true', default=RANDOM_START)
    p.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto')
    p.add_argument('--max-vram-mb', type=int, default=0, help='limite de VRAM (0 = memory growth sem limite)')
    p.add_argument('--checkpoint-every', type=int, default=100, help='episódios entre checkpoints periódicos')
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == '__main__' else parse_args([])

if ARGS.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Configurar memória da GPU ANTES de qualquer operação na GPU
gpus = tf.config.list_physical_devices('GPU')
print("=== Configuração da GPU ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs disponíveis: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")
    if ARGS.max_vram_mb > 0:
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=ARGS.max_vram_mb)])
    else:
        tf.config.experimental.set_memory_growth(gpu, True)
if ARGS.device == 'gpu' and not gpus:
    print("[ERRO] --device gpu pedido, mas o TensorFlow não encontrou GPU. "
          "Rode via ../start_training.sh (configura LD_LIBRARY_PATH) ou use --device cpu.")
    sys.exit(1)
print(f"Dispositivo de treino: {'GPU' if gpus else 'CPU'}")
print("==========================\n")

if ARGS.seed is not None:
    np.random.seed(ARGS.seed)
    tf.random.set_seed(ARGS.seed)

from dqn_agent import DQNAgent  # noqa: E402 (importar após config da GPU)
from knight_env import KnightTourEnv  # noqa: E402

BOARD_SIZE = ARGS.board_size
RANDOM_START = ARGS.random_start
EPISODES = ARGS.episodes
BATCH_SIZE = ARGS.batch_size
TARGET_UPDATE_FREQ = ARGS.target_update
REPLAY_FREQ = ARGS.replay_freq
MAX_STEPS = BOARD_SIZE * BOARD_SIZE * 3  # Teto dinâmico por tabuleiro

# Decaimento calibrado pra atingir epsilon_min (~0.01) aos 70% do treino.
# Epsilon decai por update (replay), então converte episódios -> updates
# estimados (~(N² / REPLAY_FREQ) + 1 por episódio).
_UPDATES_PER_EPISODE = max(1.0, (BOARD_SIZE * BOARD_SIZE * 0.7) / REPLAY_FREQ + 1)
EPSILON_DECAY = 0.01 ** (1.0 / (EPISODES * 0.7 * _UPDATES_PER_EPISODE))

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)


def evaluate_greedy(agent, board_size, random_start):
    """Política gulosa (sem exploração). Retorna (vitórias, média de casas, nº de inícios).

    Métrica real do modelo: vitórias durante o treino incluem exploração aleatória.
    Início no centro, ou todas as casas com solução se random_start.
    """
    eval_env = KnightTourEnv(board_size=board_size)
    if random_start:
        starts = eval_env.solvable_starts()  # N ímpar: só casas (r+c) par têm passeio
    else:
        starts = [(board_size // 2, board_size // 2)]
    saved_eps, agent.epsilon = agent.epsilon, 0.0
    wins, visited_sum = 0, 0
    for pos in starts:
        eval_env.reset()
        eval_env.board[:] = 0
        eval_env.current_pos = pos
        eval_env.board[pos] = 2
        eval_env.path = [pos]
        eval_env.visited_count = 1
        state = eval_env._get_observation()
        done = False
        while not done:
            mask = eval_env._get_valid_moves_mask()
            if not mask.any():
                break
            state, _, done, _ = eval_env.step(agent.act(state, mask))
        wins += int(eval_env.visited_count == eval_env.total_squares)
        visited_sum += eval_env.visited_count
    agent.epsilon = saved_eps
    return wins, visited_sum / len(starts), len(starts)


def main():
    run_id = time.strftime('%Y%m%d_%H%M%S')
    env = KnightTourEnv(board_size=BOARD_SIZE, random_start=RANDOM_START)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    # Cada execução em pasta própria: nunca sobrescreve checkpoints antigos
    # (models/{N}x{N}/*.weights.h5 continuam sendo os usados pelo jogo).
    board_models_dir = f'models/{BOARD_SIZE}x{BOARD_SIZE}/run_{run_id}'
    os.makedirs(board_models_dir, exist_ok=True)

    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        epsilon_decay=EPSILON_DECAY,
        memory_size=100000,
    )

    log_file = f'logs/training_log_{BOARD_SIZE}x{BOARD_SIZE}_{run_id}.csv'
    print(f"Run: {run_id}")
    print(f"  Checkpoints: RL/{board_models_dir}/")
    print(f"  Log:         RL/{log_file}")
    print(f"  Episódios: {EPISODES} | batch: {BATCH_SIZE} | replay a cada {REPLAY_FREQ} passos | "
          f"epsilon_decay/update: {EPSILON_DECAY:.6f}\n")
    log_header = ['Episode', 'Score', 'Max_Visited', 'Avg_Reward',
                  'Invalid_Moves', 'Epsilon', 'Win']
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(log_header)

    last_100 = []
    best_win_rate = 0.0
    best_greedy = (-1, -1.0)
    first_win_saved = False
    step_count = 0
    t_start = time.time()

    for e in tqdm(range(EPISODES), desc=f"Treinando {BOARD_SIZE}x{BOARD_SIZE}"):
        state = env.reset()
        total_reward = 0.0
        invalid_move_count = 0

        for t in range(MAX_STEPS):
            valid_moves_mask = env._get_valid_moves_mask()
            onward_counts = env.get_action_onward_counts()
            action = agent.act(state, valid_moves_mask, onward_counts)

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if info.get('status') == 'invalid_move':
                invalid_move_count += 1

            next_mask = env._get_valid_moves_mask()
            agent.remember(state, action, reward, next_state, done, next_mask)
            state = next_state
            step_count += 1

            # Replay frequente dentro do episódio para aprendizado mais denso
            if step_count % REPLAY_FREQ == 0:
                agent.replay(BATCH_SIZE)

            if done:
                break

        # Replay extra ao final de cada episódio
        agent.replay(BATCH_SIZE)

        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        score = t + 1
        max_visited = env.visited_count
        avg_reward = total_reward / score if score > 0 else 0.0
        win = 1 if max_visited == env.total_squares else 0

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                e, score, max_visited, f"{avg_reward:.2f}",
                invalid_move_count, f"{agent.epsilon:.4f}", win,
            ])

        last_100.append({
            'episode': e, 'score': score, 'visited': max_visited,
            'win': win, 'epsilon': agent.epsilon, 'avg_reward': avg_reward,
        })
        if len(last_100) > 100:
            last_100.pop(0)

        # Salvar só a PRIMEIRA vitória (depois que o agente aprende, quase todo
        # episódio vence — salvar todas gerava milhares de arquivos / GBs).
        if win and not first_win_saved:
            first_win_saved = True
            win_path = f"{board_models_dir}/knight_tour_dqn_b{BOARD_SIZE}_e{e}_win.weights.h5"
            agent.save(win_path)

        # Resumo + checkpoint periódico
        if (e + 1) % ARGS.checkpoint_every == 0:
            window = len(last_100)
            wins_in_group = sum(ep['win'] for ep in last_100)
            win_rate = wins_in_group / window
            avg_visited = sum(ep['visited'] for ep in last_100) / window
            best_ep = max(last_100, key=lambda x: (x['win'], x['visited']))

            print(f"\n[Eps {e + 2 - window:>5}–{e + 1:>5}] "
                  f"Vitórias: {wins_in_group:>3}/{window} | "
                  f"Média visitadas: {avg_visited:.1f}/{env.total_squares} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"{(e + 1) / (time.time() - t_start):.1f} ep/s")
            print(f"  Melhor do grupo: Ep {best_ep['episode']} | "
                  f"Visitadas: {best_ep['visited']}/{env.total_squares} | "
                  f"Win: {best_ep['win']}")

            periodic_path = f"{board_models_dir}/knight_tour_dqn_b{BOARD_SIZE}_e{e + 1}.weights.h5"
            agent.save(periodic_path)

            g_wins, g_visited, g_n = evaluate_greedy(agent, BOARD_SIZE, RANDOM_START)
            print(f"  Avaliação gulosa: {g_wins}/{g_n} vitórias | média {g_visited:.1f}/{env.total_squares} casas")
            if (g_wins, g_visited) > best_greedy:
                best_greedy = (g_wins, g_visited)
                agent.save(f"{board_models_dir}/knight_tour_dqn_b{BOARD_SIZE}_best_greedy.weights.h5")
                print("  >> Novo melhor modelo (guloso)")

            # Melhor por win rate durante o treino (inclui exploração)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = f"{board_models_dir}/knight_tour_dqn_b{BOARD_SIZE}_best.weights.h5"
                agent.save(best_path)
                print(f"  >> Novo melhor modelo! Win rate: {win_rate * 100:.0f}%")

    print(f"\nTreinamento concluído em {(time.time() - t_start) / 60:.1f} min. "
          f"Melhor win rate (treino): {best_win_rate * 100:.1f}% | "
          f"melhor guloso: {best_greedy[0]} vitórias, {best_greedy[1]:.1f} casas")
    print(f"Checkpoints em RL/{board_models_dir}/")
    agent.save(f"{board_models_dir}/knight_tour_dqn_b{BOARD_SIZE}_final.weights.h5")


if __name__ == "__main__":
    main()

# %%
