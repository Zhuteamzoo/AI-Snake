"""
Simple script to run a trained agent in the logical environment
and print its performance. This does not use your pygame UI, but
it lets you quickly sanity-check that the agent learned something.
"""

import numpy as np
import torch

from agent import Agent
from model import LinearQNet
from snake_env import SnakeGameEnv


def load_trained_agent(model_path: str = "model_best.pth") -> Agent:
    env = SnakeGameEnv()
    state_size = len(env.get_state())
    action_size = 3

    agent = Agent(state_size=state_size, action_size=action_size)
    model = LinearQNet(state_size, 128, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    agent.model = model
    return agent


def run_agent(num_episodes: int = 10, model_path: str = "model_best.pth"):
    env = SnakeGameEnv()
    agent = load_trained_agent(model_path)

    total_score = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # Greedy action selection (no exploration during evaluation)
            state0 = np.array(state, dtype=np.float32).reshape(1, -1)
            with torch.no_grad():
                q_values = agent.model(torch.tensor(state0, dtype=torch.float32))
            action = int(torch.argmax(q_values).item())

            next_state, reward, done, score = env.step(action)
            state = next_state

        total_score += score
        print(f"Episode {episode}: Score = {score}")

    avg_score = total_score / num_episodes
    print(f"Average score over {num_episodes} episodes: {avg_score:.2f}")


if __name__ == "__main__":
    run_agent(num_episodes=10, model_path="model_best.pth")

