from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class LinearQNet(nn.Module):
    """
    Small feed-forward network for Q-learning.

    - Input: state vector (e.g. 11 features).
    - Output: Q-values for each possible action.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

    def save(self, file_name: str = "model.pth"):
        """Save model weights to disk so you can load and play later."""
        model_folder_path = Path(".")
        model_folder_path.mkdir(parents=True, exist_ok=True)
        file_name = model_folder_path / file_name
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    Helper for training the Q-network.

    Encapsulates the optimizer, loss function, and one training step.
    """

    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one gradient update from a batch of experiences.

        Args are expected to be tensors or will be converted to tensors.
        """
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        # If we get a single sample, add batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predicted Q-values for current state
        pred = self.model(state)

        # Clone the predictions to build the target tensor
        target = pred.clone().detach()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_next = torch.max(self.model(next_state[idx]))
                q_new = reward[idx] + self.gamma * q_next

            # action[idx] is the index of the chosen action
            target[idx, action[idx].item()] = q_new

        # Compute loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

