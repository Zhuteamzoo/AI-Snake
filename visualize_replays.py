import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pygame


# Grid configuration (must match SnakeGameEnv)
GRID_WIDTH = 17
GRID_HEIGHT = 15
BLOCK_SIZE = 24  # rendering size per cell
PANEL_WIDTH = GRID_WIDTH * BLOCK_SIZE
PANEL_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# Overall window: 2x2 grid of panels
WINDOW_WIDTH = PANEL_WIDTH * 2
WINDOW_HEIGHT = PANEL_HEIGHT * 2
FPS = 10

REPLAYS_DIR = "replays"


@dataclass
class EpisodeReplay:
    episode: int
    steps: List[Dict]
    score: int
    total_reward: float


def load_replays_for_range(start_ep: int, end_ep: int) -> List[EpisodeReplay]:
    """Load all recorded episodes in [start_ep, end_ep] from the replays folder."""
    replays: List[EpisodeReplay] = []
    if not os.path.isdir(REPLAYS_DIR):
        return replays

    for ep in range(start_ep, end_ep + 1):
        path = os.path.join(REPLAYS_DIR, f"episode_{ep}.pkl")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        replays.append(
            EpisodeReplay(
                episode=data["episode"],
                steps=data["steps"],
                score=data["score"],
                total_reward=data["total_reward"],
            )
        )

    return replays


def performance_color(score: int, total_reward: float) -> Tuple[int, int, int]:
    """
    Map performance to a color.

    We use final score (apples eaten) as the primary metric, with simple tiers:
    - <= 0 : very bad (red)
    - 1-2  : below average (orange)
    - 3-4  : medium (yellow)
    - 5-7  : good (green)
    - >= 8 : very good (light green)
    """
    if score <= 0:
        return 255, 0, 0  # red
    if score <= 2:
        return 255, 128, 0  # orange
    if score <= 4:
        return 255, 255, 0  # yellow
    if score <= 7:
        return 0, 200, 0  # green
    return 144, 238, 144  # light green


class PanelPlayer:
    """Handles replaying a list of episodes in a rectangular panel."""

    def __init__(self, replays: List[EpisodeReplay], label: str):
        self.replays = replays
        self.label = label
        self.current_idx = 0
        self.step_idx = 0

    def current_episode(self) -> EpisodeReplay | None:
        if not self.replays:
            return None
        return self.replays[self.current_idx]

    def step(self):
        """Advance to the next frame; loop episodes when needed."""
        if not self.replays:
            return
        ep = self.replays[self.current_idx]
        self.step_idx += 1
        if self.step_idx >= len(ep.steps):
            self.current_idx = (self.current_idx + 1) % len(self.replays)
            self.step_idx = 0

    def get_frame(self):
        """Return current frame data for drawing."""
        ep = self.current_episode()
        if ep is None or not ep.steps:
            return None, None, None, None

        step = ep.steps[self.step_idx]
        snake = step["snake"]
        food = step["food"]
        score = ep.score
        color = performance_color(ep.score, ep.total_reward)
        return snake, food, score, color


def draw_panel(
    surface: pygame.Surface,
    panel: PanelPlayer,
    rect: pygame.Rect,
    font: pygame.font.Font,
):
    """Draw a single panel with its current episode frame and label."""
    pygame.draw.rect(surface, (30, 30, 30), rect)

    snake, food, score, color = panel.get_frame()
    if snake is None or food is None:
        # No data – draw placeholder text
        text = font.render("No replays found", True, (255, 255, 255))
        surface.blit(text, (rect.x + 10, rect.y + 10))
        label_text = font.render(panel.label, True, (200, 200, 200))
        surface.blit(label_text, (rect.x + 10, rect.y + 30))
        return

    # Draw grid background
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            cell_rect = pygame.Rect(
                rect.x + x * BLOCK_SIZE,
                rect.y + y * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE,
            )
            pygame.draw.rect(surface, (50, 100, 50), cell_rect)

    # Draw food
    fx, fy = food
    food_rect = pygame.Rect(
        rect.x + fx * BLOCK_SIZE,
        rect.y + fy * BLOCK_SIZE,
        BLOCK_SIZE,
        BLOCK_SIZE,
    )
    pygame.draw.rect(surface, (255, 0, 0), food_rect)

    # Draw snake
    for i, (sx, sy) in enumerate(snake):
        seg_rect = pygame.Rect(
            rect.x + sx * BLOCK_SIZE,
            rect.y + sy * BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
        )
        # Head a bit brighter
        if i == 0:
            head_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(surface, head_color, seg_rect)
        else:
            pygame.draw.rect(surface, color, seg_rect)

    # Episode/score label
    ep = panel.current_episode()
    label_line1 = f"{panel.label}"
    label_line2 = f"Ep {ep.episode} | Score {ep.score}"
    text1 = font.render(label_line1, True, (255, 255, 255))
    text2 = font.render(label_line2, True, (220, 220, 220))
    surface.blit(text1, (rect.x + 5, rect.y + 5))
    surface.blit(text2, (rect.x + 5, rect.y + 25))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake AI Replay Dashboard")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Load replays for different ranges
    early = load_replays_for_range(1, 10)
    mid1 = load_replays_for_range(250, 260)
    mid2 = load_replays_for_range(500, 510)
    late = load_replays_for_range(750, 760)

    panels = [
        PanelPlayer(early, "Episodes 1–10"),
        PanelPlayer(mid1, "Episodes 250–260"),
        PanelPlayer(mid2, "Episodes 500–510"),
        PanelPlayer(late, "Episodes 750–760"),
    ]

    panel_rects = [
        pygame.Rect(0, 0, PANEL_WIDTH, PANEL_HEIGHT),
        pygame.Rect(PANEL_WIDTH, 0, PANEL_WIDTH, PANEL_HEIGHT),
        pygame.Rect(0, PANEL_HEIGHT, PANEL_WIDTH, PANEL_HEIGHT),
        pygame.Rect(PANEL_WIDTH, PANEL_HEIGHT, PANEL_WIDTH, PANEL_HEIGHT),
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        for panel, rect in zip(panels, panel_rects):
            draw_panel(screen, panel, rect, font)
            panel.step()

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

