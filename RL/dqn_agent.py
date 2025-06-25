import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    """
    Agente de Deep Q-Learning (DQN) para o problema do Passeio do Cavalo.
    """
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        
        # Hiperparâmetros
        self.gamma = gamma    # Fator de desconto
        self.epsilon = epsilon  # Taxa de exploração (greedy)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Constrói uma rede neural convolucional (CNN) para o modelo DQN.
        CNNs são mais eficazes para processar dados espaciais como um tabuleiro.
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                         input_shape=self.state_shape, padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """ Copia os pesos do modelo principal para o modelo alvo. """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """ Armazena a experiência na memória de replay. """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves_mask):
        """
        Escolhe uma ação, usando a máscara para garantir que seja válida.
        """
        # Exploração: escolhe uma ação válida aleatoriamente
        if np.random.rand() <= self.epsilon:
            valid_indices = np.where(valid_moves_mask)[0]
            if len(valid_indices) > 0:
                return random.choice(valid_indices)
            else:
                return random.randrange(self.action_size) # Fallback

        # Explotação: escolhe a melhor ação válida com base nos Q-valores
        state_tensor = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state_tensor, verbose=0)[0]
        
        # Aplica a máscara: define Q-valores de ações inválidas como -infinito
        masked_act_values = np.where(valid_moves_mask, act_values, -np.inf)
        
        return np.argmax(masked_act_values)

    def replay(self, batch_size):
        """
        Treina o agente com experiências da memória (Experience Replay).
        """
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Alvo para o treinamento do batch
        target_q = self.model.predict(states, verbose=0)
        # Q-valores futuros para o cálculo do TD target
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        self.model.fit(states, target_q, epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """ Carrega os pesos do modelo a partir de um arquivo. """
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        """ Salva os pesos do modelo em um arquivo. """
        self.model.save_weights(name) 