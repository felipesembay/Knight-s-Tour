import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def _make_optimizer(learning_rate):
    """Adam sem XLA.

    No TF 2.13 (tf_env) o Adam compila com XLA por padrão e falha na GPU com
    'libdevice not found' (docs/TECHNICAL_AUDIT.md, A2). Keras 3 (TF >= 2.16)
    não aceita o argumento jit_compile, então cai no construtor padrão.
    """
    try:
        return Adam(learning_rate=learning_rate, clipnorm=1.0, jit_compile=False)
    except (TypeError, ValueError):  # Keras 3: "Argument(s) not recognized"
        return Adam(learning_rate=learning_rate, clipnorm=1.0)


class ReplayBuffer:
    """Buffer circular pré-alocado (evita montar arrays a partir de tuplas a cada batch)."""

    def __init__(self, capacity, state_shape, action_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.ones((capacity, action_size), dtype=bool)
        self.size = 0
        self.ptr = 0

    def add(self, state, action, reward, next_state, done, next_mask):
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.next_masks[i] = next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx], self.next_masks[idx])

    def __len__(self):
        return self.size


class DQNAgent:
    """Agente Double DQN para o problema do Passeio do Cavalo.

    Arquitetura da rede inalterada -> checkpoints .weights.h5 antigos continuam
    carregando. Mudanças de desempenho/correção:
      - train step compilado com tf.function (~45x mais rápido que predict+fit);
      - action masking também no alvo Double DQN (próximo estado);
      - otimizador sem XLA (GPU funcional no TF 2.13).
    """

    def __init__(self, state_shape, action_size, learning_rate=0.0005, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.9993, epsilon_min=0.01, memory_size=100000):
        self.state_shape = tuple(state_shape)
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_size, self.state_shape, action_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.optimizer = _make_optimizer(learning_rate)
        self.loss_fn = tf.keras.losses.Huber()
        self._train_step = tf.function(self._train_step_impl)
        self._q_values = tf.function(lambda x: self.model(x, training=False))

    def _build_model(self):
        model = Sequential([
            Input(shape=self.state_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, next_mask=None):
        """next_mask: ações válidas em next_state. None = todas (comportamento antigo)."""
        if next_mask is None:
            next_mask = np.ones(self.action_size, dtype=bool)
        self.memory.add(state, action, reward, next_state, done, next_mask)

    def q_values(self, state):
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
        return self._q_values(state_tensor).numpy()[0]

    def act(self, state, valid_moves_mask, onward_counts=None):
        if np.random.rand() <= self.epsilon:
            valid_indices = np.where(valid_moves_mask)[0]
            if len(valid_indices) == 0:
                return int(np.random.randint(self.action_size))

            if onward_counts is not None:
                # Exploração guiada por Warnsdorff: prioriza ações que deixam
                # menos movimentos futuros (menor chance de prender o agente),
                # mas sem ser 100% greedy — ainda dá chance a outras opções.
                valid_onward = onward_counts[valid_indices].astype(np.float32)
                weights = 1.0 / (valid_onward + 1.0) ** 2
                weights /= weights.sum()
                return int(np.random.choice(valid_indices, p=weights))

            return int(np.random.choice(valid_indices))

        act_values = self.q_values(state)
        masked_act_values = np.where(valid_moves_mask, act_values, -np.inf)
        return int(np.argmax(masked_act_values))

    def _train_step_impl(self, states, actions, rewards, next_states, dones, next_masks):
        # Double DQN: online escolhe a ação (só entre válidas), target avalia.
        next_q_online = self.model(next_states, training=False)
        next_q_online = tf.where(next_masks, next_q_online, tf.fill(tf.shape(next_q_online), -1e9))
        best_next = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        next_q_target = self.target_model(next_states, training=False)
        next_values = tf.gather(next_q_target, best_next, batch_dims=1)
        has_valid = tf.cast(tf.reduce_any(next_masks, axis=1), tf.float32)
        targets = rewards + self.gamma * next_values * (1.0 - dones) * has_valid
        targets = tf.stop_gradient(targets)

        with tf.GradientTape() as tape:
            q = self.model(states, training=True)
            q_taken = tf.gather(q, actions, batch_dims=1)
            loss = self.loss_fn(targets, q_taken)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)
        loss = self._train_step(*(tf.convert_to_tensor(x) for x in batch))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        try:
            self.model.load_weights(name)
        except Exception as e:
            # .weights.h5 gravado por outra versão do Keras (ex.: TF 2.13 / Keras 2
            # no tf_env, lido pelo Keras 3 do .venv) usa nomes de camada diferentes.
            if not self._load_weights_by_shape(name):
                raise e
        self.update_target_model()

    def _load_weights_by_shape(self, path):
        """Carrega um .weights.h5 casando cada camada pelo formato dos pesos.

        Nesta rede todos os formatos de camada são distintos, então o casamento
        é único. Retorna False se não conseguir casar todas as camadas.
        """
        import re

        import h5py

        groups = {}
        with h5py.File(path, 'r') as f:
            def visit(key, obj):
                m = re.match(r'(.*)/vars/(\d+)$', key)
                if m and isinstance(obj, h5py.Dataset):
                    groups.setdefault(m.group(1), {})[int(m.group(2))] = obj[()]
            f.visititems(visit)
        candidates = [[g[i] for i in sorted(g)] for g in groups.values()]

        assignments = []
        for layer in self.model.layers:
            expected = [tuple(w.shape) for w in layer.weights]
            if not expected:
                continue
            match = next((c for c in candidates if [a.shape for a in c] == expected), None)
            if match is None:
                return False
            candidates.remove(match)
            assignments.append((layer, match))
        for layer, weights in assignments:
            layer.set_weights(weights)
        return True

    def save(self, name):
        self.model.save_weights(name)
