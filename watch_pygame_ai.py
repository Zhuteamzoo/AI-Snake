import sys

import numpy as np
import pygame
import torch

from model import LinearQNet
from snake_env import SnakeGameEnv


# Visual settings
GRID_WIDTH = 17
GRID_HEIGHT = 15
BLOCK_SIZE = 32  # purely for rendering
WIDTH = GRID_WIDTH * BLOCK_SIZE
HEIGHT = GRID_HEIGHT * BLOCK_SIZE
FPS = 10  # speed of the visual game


def load_model(model_path: str = "model_best.pth") -> LinearQNet:
    """Load the trained Q-network from disk."""
    state_size = 11  # must match SnakeGameEnv.get_state() length
    action_size = 3  # [straight, right, left]
    hidden_size = 128

    model = LinearQNet(state_size, hidden_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def get_action_from_model(model: LinearQNet, state):
    """Greedy action selection from the trained model (no exploration)."""
    state0 = np.array(state, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        q_values = model(torch.tensor(state0, dtype=torch.float32))
    action = int(torch.argmax(q_values).item())
    return action


def draw_game(screen, env: SnakeGameEnv, font):
    """Draw the current SnakeGameEnv state to the pygame screen."""
    # Background
    screen.fill((72, 168, 50))  # green

    # Draw food
    food = env.food
    pygame.draw.rect(
        screen,
        (255, 0, 0),
        pygame.Rect(food.x * BLOCK_SIZE, food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
    )

    # Draw snake
    for i, segment in enumerate(env.snake):
        color = (0, 0, 255) if i == 0 else (0, 150, 255)
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(
                segment.x * BLOCK_SIZE,
                segment.y * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE,
            ),
        )

    # Score text
    score_text = font.render(f"Score: {env.score}", True, (255, 255, 255))
    screen.blit(score_text, (5, 5))


def watch(num_episodes: int = 5, model_path: str = "model_best.pth"):
    """Open a pygame window and watch the trained model play Snake."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake AI - Watch Trained Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)

    env = SnakeGameEnv(width=GRID_WIDTH, height=GRID_HEIGHT)
    model = load_model(model_path)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            # Handle window events so you can close the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Choose action and step the environment
            action = get_action_from_model(model, state)
            next_state, reward, done, score = env.step(action)
            state = next_state

            # Draw current state
            draw_game(screen, env, font)
            pygame.display.flip()

            clock.tick(FPS)

        print(f"Episode {episode}: Score = {env.score}")

    pygame.quit()


if __name__ == "__main__":
    watch(num_episodes=5, model_path="model_best.pth")

