import os
import pickle
import time

import numpy as np

from agent import Agent
from plot_utils import plot
from snake_env import SnakeGameEnv


def train(
    episodes: int = 1000,
    render_every: int = 0,
):
    """
    Main training loop for the Snake DQN agent.

    Args:
        episodes: how many games (episodes) to train for.
        render_every: not used in this non-visual env, but kept for future
                      integration with a visual version.
    """
    env = SnakeGameEnv()
    state_size = len(env.get_state())
    action_size = 3  # [straight, turn right, turn left]

    agent = Agent(state_size=state_size, action_size=action_size)

    scores = []
    mean_scores = []
    record = 0

    # Ensure directory for episode replays exists
    replays_dir = "replays"
    os.makedirs(replays_dir, exist_ok=True)

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        # For selected episode ranges, record frames for later visualization.
        record_this_episode = (
            1 <= episode <= 10
            or 250 <= episode <= 260
            or 500 <= episode <= 510
            or 750 <= episode <= 760
        )
        episode_steps = []

        while not done:
            action = agent.get_action(state)

            next_state, reward, done, score = env.step(action)

            loss = agent.train_short_memory(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)

            agent.remember(state, action, reward, next_state, done)

            # If we are recording this episode, store a snapshot
            if record_this_episode:
                # env.snake is a list of Points; convert to plain tuples
                snake_snapshot = [(pt.x, pt.y) for pt in env.snake]
                food_snapshot = (env.food.x, env.food.y)
                episode_steps.append(
                    {
                        "snake": snake_snapshot,
                        "food": food_snapshot,
                    }
                )

            state = next_state
            total_reward += reward

        # Episode finished – train on a batch from replay memory
        batch_loss = agent.train_long_memory()

        # Save replay for this episode if requested
        if record_this_episode:
            replay_path = os.path.join(replays_dir, f"episode_{episode}.pkl")
            with open(replay_path, "wb") as f:
                pickle.dump(
                    {
                        "episode": episode,
                        "steps": episode_steps,
                        "score": score,
                        "total_reward": total_reward,
                    },
                    f,
                )

        agent.n_games += 1
        scores.append(score)
        mean_score = np.mean(scores[-50:])
        mean_scores.append(mean_score)

        if score > record:
            record = score
            agent.model.save("model_best.pth")

        epsilon = agent._get_epsilon()
        avg_loss = float(np.mean(losses)) if losses else 0.0

        print(
            f"Episode {episode:4d} | "
            f"Score: {score:3d} | "
            f"TotalReward: {total_reward:7.2f} | "
            f"Record: {record:3d} | "
            f"MeanScore(50): {mean_score:5.2f} | "
            f"Epsilon: {epsilon:5.3f} | "
            f"Loss: {avg_loss:7.4f}"
        )

        # Update live plot
        plot(scores, mean_scores)

    # Save final model at the end of training
    agent.model.save("model_final.pth")
    print("Training finished. Models saved as 'model_best.pth' and 'model_final.pth'.")


if __name__ == "__main__":
    train(episodes=1000)

