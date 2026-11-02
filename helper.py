import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot(scores, mean_scores):

    plt.figure(figsize=(10, 6))

    games = range(1, len(scores) + 1)

    plt.plot(
        games,
        scores,
        label="Game Score",
        alpha=0.4
    )

    plt.plot(
        games,
        mean_scores,
        label="Average Score",
        linewidth=2
    )

    # 20-game moving average
    window = 20

    if len(scores) >= window:

        moving_avg = [
            sum(scores[i-window:i]) / window
            for i in range(window, len(scores) + 1)
        ]

        plt.plot(
            range(window, len(scores) + 1),
            moving_avg,
            label="20-Game Moving Average",
            linewidth=2
        )

    plt.title("Snake AI - Training Performance")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    # Save graph instead of opening another GUI window
    plt.savefig("training_progress.png", dpi=150)

    plt.close()