import random
from collections import namedtuple
from math import inf

import numpy as np


Point = namedtuple("Point", "x y")


class Direction:
    """Grid directions used by the environment."""

    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class SnakeGameEnv:
    """
    Simple, non-visual Snake environment suitable for reinforcement learning.

    - Grid-based, no direct dependency on pygame.
    - State is a compact feature vector (not pixels).
    - Actions: [straight, turn right, turn left] relative to current direction.
    """

    def __init__(self, width: int = 17, height: int = 15):
        self.width = width
        self.height = height

        # Reward configuration – all in one place for easy tuning.
        # These values are chosen to strongly encourage reaching the apple,
        # mildly penalize wasting time, and clearly discourage moving
        # away from the apple or dying.
        self.reward_apple = 10.0      # main positive signal
        self.reward_death = -10.0     # strong negative for hitting wall/self
        self.reward_step = -0.02      # small penalty each move
        self.reward_closer = 0.15     # bonus if distance to apple decreases
        self.reward_farther = -0.15   # penalty if distance to apple increases

        # Max steps allowed without eating an apple; prevents endless loops.
        # On timeout we end the episode and apply an extra penalty.
        self.max_steps_without_apple = self.width * self.height
        self.reward_timeout = -5.0

        self.reset()

    # ------------------------------------------------------------------
    # Public API expected by the training loop
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the game to the initial state and return the initial observation."""
        # Start roughly in the center of the grid
        center_x = self.width // 2
        center_y = self.height // 2
        self.direction = Direction.RIGHT

        self.head = Point(center_x, center_y)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
        ]

        self.score = 0
        self.steps_since_last_apple = 0
        self._place_food()

        # Used for distance-based reward shaping
        self.prev_distance_to_food = self._distance_to_food()

        return self.get_state()

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action: one of [0, 1, 2] meaning [straight, turn right, turn left].
                    This encoding is orientation-invariant and tends to train
                    more stably than absolute [up, down, left, right].

        Returns:
            next_state (np.ndarray),
            reward (float),
            done (bool),
            score (int)
        """
        self.steps_since_last_apple += 1

        # 1. Update direction based on relative action
        self._move_direction(action)

        # 2. Move snake head
        self._move_head()
        self.snake.insert(0, self.head)

        # 3. Check for terminal conditions
        reward = 0.0
        done = False

        if self._is_collision(self.head):
            reward += self.reward_death
            done = True
            return self.get_state(), reward, done, self.score

        # 4. Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward += self.reward_apple
            self._place_food()
            self.prev_distance_to_food = self._distance_to_food()
        else:
            self.snake.pop()

        # 5. Reward shaping based on distance to food
        distance = self._distance_to_food()
        if distance < self.prev_distance_to_food:
            reward += self.reward_closer
        elif distance > self.prev_distance_to_food:
            reward += self.reward_farther
        self.prev_distance_to_food = distance

        # Small time penalty to encourage efficiency
        reward += self.reward_step

        # Optional safety: end episode if snake wanders too long without progress.
        # This helps break degenerate looping behavior.
        if self.steps_since_last_apple > self.max_steps_without_apple:
            reward += self.reward_timeout
            done = True

        return self.get_state(), reward, done, self.score

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------

    def get_state(self):
        """
        Compute a compact numerical state vector.

        State features (11 values):
        - danger straight, danger right, danger left
        - direction up, direction right, direction down, direction left
        - food up, food right, food down, food left
        """
        head = self.head

        # Points in the three potential directions relative to current motion
        point_l = None
        point_r = None
        point_s = None

        if self.direction == Direction.RIGHT:
            point_s = Point(head.x + 1, head.y)
            point_r = Point(head.x, head.y + 1)
            point_l = Point(head.x, head.y - 1)
        elif self.direction == Direction.LEFT:
            point_s = Point(head.x - 1, head.y)
            point_r = Point(head.x, head.y - 1)
            point_l = Point(head.x, head.y + 1)
        elif self.direction == Direction.UP:
            point_s = Point(head.x, head.y - 1)
            point_r = Point(head.x + 1, head.y)
            point_l = Point(head.x - 1, head.y)
        else:  # DOWN
            point_s = Point(head.x, head.y + 1)
            point_r = Point(head.x - 1, head.y)
            point_l = Point(head.x + 1, head.y)

        danger_straight = self._is_collision(point_s)
        danger_right = self._is_collision(point_r)
        danger_left = self._is_collision(point_l)

        dir_up = self.direction == Direction.UP
        dir_right = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT

        food_up = self.food.y < head.y
        food_right = self.food.x > head.x
        food_down = self.food.y > head.y
        food_left = self.food.x < head.x

        state = np.array(
            [
                int(danger_straight),
                int(danger_right),
                int(danger_left),
                int(dir_up),
                int(dir_right),
                int(dir_down),
                int(dir_left),
                int(food_up),
                int(food_right),
                int(food_down),
                int(food_left),
            ],
            dtype=np.float32,
        )

        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_direction(self, action: int):
        """Update self.direction based on relative action [straight, right, left]."""
        assert action in (0, 1, 2), "Action must be 0, 1, or 2"

        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)

        if action == 1:  # turn right
            idx = (idx + 1) % 4
        elif action == 2:  # turn left
            idx = (idx - 1) % 4

        self.direction = directions[idx]

    def _move_head(self):
        """Move the snake's head one cell in the current direction."""
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.UP:
            y -= 1
        else:  # DOWN
            y += 1
        self.head = Point(x, y)

    def _place_food(self):
        """Place food at a random empty location on the grid."""
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            food = Point(x, y)
            if food not in self.snake:
                self.food = food
                self.steps_since_last_apple = 0
                return

    def _is_collision(self, pt: Point):
        """Return True if the given point collides with the wall or the snake body."""
        if pt is None:
            return True

        # Wall collision
        if pt.x < 0 or pt.x >= self.width or pt.y < 0 or pt.y >= self.height:
            return True

        # Self collision (ignore head at index 0)
        if pt in self.snake[1:]:
            return True

        return False

    def _distance_to_food(self):
        """Manhattan distance from the snake's head to the food."""
        if self.food is None or self.head is None:
            return inf
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

