import gym
from gym import spaces
import numpy as np

class KnightTourEnv(gym.Env):
    """
    Ambiente customizado para o problema do Passeio do Cavalo (Knight's Tour)
    seguindo a interface do OpenAI Gym.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, board_size=5, random_start=False):
        super(KnightTourEnv, self).__init__()

        self.board_size = board_size
        self.total_squares = board_size * board_size
        self.random_start = random_start

        self.knight_moves = np.array([
            [-2, -1], [-2, 1], [-1, -2], [-1, 2],
            [1, -2], [1, 2], [2, -1], [2, 1]
        ])

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(board_size, board_size, 3),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)

        if self.random_start:
            starts = self.solvable_starts()
            self.current_pos = starts[np.random.randint(len(starts))]
        else:
            center = self.board_size // 2
            self.current_pos = (center, center)

        self.board[self.current_pos] = 2
        self.path = [self.current_pos]
        self.visited_count = 1

        return self._get_observation()

    def solvable_starts(self):
        """Casas iniciais a partir das quais existe passeio aberto.

        Em tabuleiro ímpar (5x5, 7x7, 9x9) há uma casa a mais da cor (r+c) par;
        como o cavalo alterna cores, o passeio só existe começando numa casa par.
        Em tabuleiro par (6x6, 8x8) todas as casas servem.
        """
        n = self.board_size
        return [(r, c) for r in range(n) for c in range(n)
                if n % 2 == 0 or (r + c) % 2 == 0]

    def _get_observation(self):
        pos_channel = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        pos_channel[self.current_pos] = 1.0

        visited_channel = (self.board == 1).astype(np.float32)
        valid_moves_channel = self._get_valid_moves_board()

        return np.stack([pos_channel, visited_channel, valid_moves_channel], axis=-1)

    def _count_onward_moves_after(self, new_pos):
        """
        Conta movimentos válidos a partir de new_pos após executar o movimento
        (heurística de Warnsdorff). Considera que current_pos ficará visitado.
        """
        count = 0
        for move in self.knight_moves:
            r, c = new_pos[0] + move[0], new_pos[1] + move[1]
            if (0 <= r < self.board_size and 0 <= c < self.board_size
                    and self.board[r, c] == 0
                    and (r, c) != self.current_pos):  # current_pos ficará visitado
                count += 1
        return count

    def get_action_onward_counts(self):
        """
        Para cada uma das 8 ações, quantos movimentos válidos sobram depois
        dela (heurística de Warnsdorff). Ações inválidas retornam -1.
        Usado pra guiar a exploração aleatória do agente para o mesmo
        heurístico que orienta a recompensa.
        """
        counts = np.full(8, -1, dtype=np.int8)
        for i, move in enumerate(self.knight_moves):
            new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
            if self._is_valid_move(new_pos):
                counts[i] = self._count_onward_moves_after(new_pos)
        return counts

    def step(self, action):
        move = self.knight_moves[action]
        new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])

        done = False
        info = {}

        if not self._is_valid_move(new_pos):
            reward = -5.0
            info['status'] = 'invalid_move'
            return self._get_observation(), reward, done, info

        # Calcular Warnsdorff ANTES de mover para contagem correta
        onward = self._count_onward_moves_after(new_pos)

        # Executar movimento
        self.board[self.current_pos] = 1
        self.current_pos = new_pos
        self.board[self.current_pos] = 2
        self.path.append(self.current_pos)
        self.visited_count += 1

        if self.visited_count == self.total_squares:
            # Recompensa alta por completar o passeio
            reward = 50.0
            done = True
            info['status'] = 'win'

        elif not self._has_valid_moves():
            # Penalidade fixa alta + proporcional ao quanto falta: travar
            # sempre pesa mais que qualquer soma de recompensas por passo.
            remaining = self.total_squares - self.visited_count
            reward = -(10.0 + remaining)
            done = True
            info['status'] = 'stuck'

        else:
            # Recompensa quase-constante por passo válido (não escala com
            # progresso, pra não recompensar avançar por avançar).
            reward = 0.1

            # Bônus de Warnsdorff: prefere casas com MENOS movimentos futuros
            # Valores baixos de onward = melhor escolha estratégica
            warnsdorff_bonus = max(0.0, (5 - onward)) * 0.05
            reward += warnsdorff_bonus

        return self._get_observation(), reward, done, info

    def _is_valid_move(self, pos):
        row, col = pos
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        if self.board[row, col] != 0:
            return False
        return True

    def _get_valid_moves_mask(self):
        mask = np.zeros(8, dtype=bool)
        for i, move in enumerate(self.knight_moves):
            new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
            if self._is_valid_move(new_pos):
                mask[i] = True
        return mask

    def _get_valid_moves_board(self):
        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for move in self.knight_moves[self._get_valid_moves_mask()]:
            pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
            board[pos] = 1.0
        return board

    def _has_valid_moves(self):
        return self._get_valid_moves_mask().any()

    def render(self, mode='console'):
        if mode == 'console':
            render_board = self.board.copy().astype(str)
            render_board[render_board == '0'] = '.'
            render_board[render_board == '1'] = 'V'
            render_board[self.current_pos] = 'K'
            print(f"Visited: {self.visited_count}/{self.total_squares}")
            print("\n".join(" ".join(row) for row in render_board))
            print("-" * 20)
