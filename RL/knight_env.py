import gym
from gym import spaces
import numpy as np

class KnightTourEnv(gym.Env):
    """
    Ambiente customizado para o problema do Passeio do Cavalo (Knight's Tour)
    seguindo a interface do OpenAI Gym.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, board_size=5):
        super(KnightTourEnv, self).__init__()

        self.board_size = board_size
        self.total_squares = self.board_size * self.board_size
        
        self.knight_moves = np.array([
            [-2, -1], [-2, 1], [-1, -2], [-1, 2],
            [1, -2], [1, 2], [2, -1], [2, 1]
        ])
        
        self.action_space = spaces.Discrete(8)

        # Representação de estado com 3 canais (posição atual, visitados, movimentos válidos)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.board_size, self.board_size, 3), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        """
        Reseta o ambiente para um novo episódio.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        
        # Posição inicial (fixa em (0,0) para consistência ou aleatória)
        #self.current_pos = (0, 0)
        center = self.board_size // 2
        self.current_pos = (center, center)
        # self.current_pos = (np.random.randint(0, self.board_size), np.random.randint(0, self.board_size))

        self.board[self.current_pos] = 2
        
        self.path = [self.current_pos]
        self.visited_count = 1
        
        return self._get_observation()

    def _get_observation(self):
        pos_channel = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        pos_channel[self.current_pos] = 1.0

        visited_channel = (self.board == 1).astype(np.float32)
        
        valid_moves_channel = self._get_valid_moves_board()

        return np.stack([pos_channel, visited_channel, valid_moves_channel], axis=-1)

    def step(self, action):
        """
        Executa uma ação no ambiente.
        """
        move = self.knight_moves[action]
        new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])

        done = False
        info = {}

        if not self._is_valid_move(new_pos):
            # Penalidade por movimento inválido (fora do tabuleiro ou já visitado)
            # O episódio NÃO termina, dando ao agente a chance de tentar outra ação.
            reward = -2.0
            info['status'] = 'invalid_move'
        else:
            # Recompensa crescente com base no progresso
            #reward = 1.0 + (self.visited_count / self.total_squares)
            reward = 1.0 + (self.visited_count / self.total_squares) * 5.0
            
            self.board[self.current_pos] = 1
            self.current_pos = new_pos
            self.board[self.current_pos] = 2
            
            self.path.append(self.current_pos)
            self.visited_count += 1

            if self.visited_count == self.total_squares:
                # Grande recompensa por completar o passeio com sucesso
                reward = -1.0 * (self.total_squares - self.visited_count)
                done = True
                info['status'] = 'win'
            elif not self._has_valid_moves():
                # Penalidade por ficar preso sem movimentos válidos
                reward = -10.0
                done = True
                info['status'] = 'stuck'

        return self._get_observation(), reward, done, info

    def _is_valid_move(self, pos):
        """ Verifica se um movimento é válido. """
        row, col = pos
        # Verifica se está dentro do tabuleiro
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        # Verifica se a casa já foi visitada
        if self.board[row, col] != 0: # Não pode ir para casa visitada ou atual
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
        valid_actions = self.knight_moves[self._get_valid_moves_mask()]
        
        for move in valid_actions:
            pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
            board[pos] = 1.0
        return board
        
    def _has_valid_moves(self):
        """ Verifica se existem movimentos válidos a partir da posição atual. """
        return self._get_valid_moves_mask().any()

    def render(self, mode='console'):
        """ Renderiza o estado atual do ambiente. """
        if mode == 'console':
            render_board = self.board.copy().astype(str)
            render_board[render_board == '0'] = '.'
            render_board[render_board == '1'] = 'V'
            render_board[self.current_pos] = 'K'
            print(f"Visited: {self.visited_count}/{self.total_squares}")
            print("\n".join(" ".join(row) for row in render_board))
            print("-" * 20) 