import sys

import pygame

from snake_env import Direction, SnakeGameEnv


# Visual configuration – must match the training environment grid
GRID_WIDTH = 17
GRID_HEIGHT = 15
BLOCK_SIZE = 32  # purely for rendering
WIDTH = GRID_WIDTH * BLOCK_SIZE
HEIGHT = GRID_HEIGHT * BLOCK_SIZE
FPS = 10


def direction_from_key(key):
    """Map arrow key to absolute Direction value."""
    if key == pygame.K_RIGHT:
        return Direction.RIGHT
    if key == pygame.K_LEFT:
        return Direction.LEFT
    if key == pygame.K_UP:
        return Direction.UP
    if key == pygame.K_DOWN:
        return Direction.DOWN
    return None


def relative_action(current_dir, desired_dir):
    """
    Given the current direction and a desired absolute direction,
    return the relative action expected by SnakeGameEnv:
    0 = straight, 1 = turn right, 2 = turn left.
    """
    if desired_dir is None:
        return 0

    directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx_cur = directions.index(current_dir)
    idx_des = directions.index(desired_dir)

    if idx_des == idx_cur:
        return 0  # straight
    if (idx_cur + 1) % 4 == idx_des:
        return 1  # right
    if (idx_cur - 1) % 4 == idx_des:
        return 2  # left

    # Opposite direction: treat as two left turns (just go left once).
    return 2


def draw_grid_background(screen):
    """Draw a simple checkerboard-style background."""
    colors = [(72, 168, 50), (67, 160, 45)]
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            color = colors[(x + y) % 2]
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, color, rect)


def draw_game(screen, env, font):
    """Render the current SnakeGameEnv state."""
    draw_grid_background(screen)

    # Draw food
    fx, fy = env.food.x, env.food.y
    food_rect = pygame.Rect(fx * BLOCK_SIZE, fy * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, (255, 0, 0), food_rect)

    # Draw snake (head in a brighter color)
    for i, seg in enumerate(env.snake):
        color = (0, 0, 255) if i == 0 else (0, 150, 255)
        rect = pygame.Rect(seg.x * BLOCK_SIZE, seg.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, color, rect)

    # Score text (apples eaten)
    score_text = font.render(f"Score: {env.score}", True, (255, 255, 255))
    screen.blit(score_text, (5, 5))


def main():
    """
    Human-playable Snake game that uses the exact same
    SnakeGameEnv logic that the RL agent trains on.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Game (Same as RL Environment)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)

    env = SnakeGameEnv(width=GRID_WIDTH, height=GRID_HEIGHT)
    state = env.reset()
    done = False

    # Default action: go straight in the current direction
    pending_action = 0
    # Wait for the first arrow-key press before moving
    waiting_for_first_move = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                # Map arrow keys to a relative action
                desired_dir = direction_from_key(event.key)
                if desired_dir is not None:
                    pending_action = relative_action(env.direction, desired_dir)
                    waiting_for_first_move = False

        if not done and not waiting_for_first_move:
            # Step the environment using the last chosen action
            next_state, reward, done, score = env.step(pending_action)
            state = next_state
            # After applying the action once, default back to straight
            pending_action = 0

        # If the episode ended, reset upon pressing any key
        if done:
            # Simple game-over text
            draw_game(screen, env, font)
            over_text = font.render(
                "Game Over - Press any key to restart, ESC to quit", True, (255, 255, 255)
            )
            screen.blit(over_text, (10, HEIGHT // 2))
            pygame.display.flip()

            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                        # Restart
                        env = SnakeGameEnv(width=GRID_WIDTH, height=GRID_HEIGHT)
                        state = env.reset()
                        done = False
                        pending_action = 0
                        waiting_for_first_move = True
                        waiting = False
                clock.tick(FPS)
            continue

        # Draw current frame
        draw_game(screen, env, font)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
