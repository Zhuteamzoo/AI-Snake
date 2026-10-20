import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch

from model import LinearQNet, QTrainer


class Agent:
    """
    Deep Q-learning agent with experience replay and epsilon-greedy exploration.

    - Chooses actions given a state.
    - Stores experiences in replay memory.
    - Trains a neural network to approximate Q-values.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        max_memory: int = 100_000,
        batch_size: int = 1_000,
        gamma: float = 0.9,
        lr: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.001,
        hidden_size: int = 128,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # Epsilon-greedy exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.n_games = 0
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=max_memory
        )

        self.model = LinearQNet(state_size, hidden_size, action_size)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)

    # ------------------------------------------------------------------
    # Epsilon-greedy policy
    # ------------------------------------------------------------------

    def _get_epsilon(self) -> float:
        """
        Compute current epsilon based on how many games have been played.

        Uses an exponential decay down to epsilon_end.
        """
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.epsilon_decay * self.n_games
        )
        return float(epsilon)

    def get_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy exploration.

        Returns:
            action index (0, 1, or 2)
        """
        epsilon = self._get_epsilon()

        # Exploration
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation (use the model)
        state0 = np.array(state, dtype=np.float32).reshape(1, -1)
        state_tensor = torch.tensor(state0, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store an experience tuple in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_long_memory(self):
        """
        Train from a random batch of experiences.

        This is where experience replay happens.
        """
        if len(self.memory) < self.batch_size:
            mini_sample = list(self.memory)
        else:
            mini_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Train from a single step (online update)."""
        loss = self.trainer.train_step(
            np.array(state, dtype=np.float32),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            done,
        )
        return loss

