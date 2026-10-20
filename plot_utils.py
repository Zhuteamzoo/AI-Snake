from typing import Iterable, List

import matplotlib.pyplot as plt
from IPython import display


plt.ion()


def plot(scores: Iterable[int], mean_scores: Iterable[float]):
    """
    Live plot of scores and moving average during training.

    This is optional but helps you see if the agent is improving.
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    scores = list(scores)
    mean_scores = list(mean_scores)

    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.legend()
    plt.grid(True)

    plt.pause(0.001)

