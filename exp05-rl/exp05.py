import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ------------------------------------------------------------
# EXPERIMENT 5: REINFORCEMENT LEARNING USING Q-LEARNING
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

for folder in (FIG_DIR, TABLE_DIR):
    os.makedirs(folder, exist_ok=True)

ROWS, COLS = 5, 7
START = (4, 0)
GOAL = (0, 6)
PIT = (2, 6)

WALLS = {
    (1, 2), (2, 2), (3, 2),
    (1, 4), (2, 4)
}

ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}
ACTION_NAMES = ["U", "D", "L", "R"]

ALPHA = 0.5
GAMMA = 0.95
EPISODES = 600
MAX_STEPS = 120
EPSILONS = [0.05, 0.2, 0.5]
SEED = 42


def valid_state(state):
    r, c = state
    return (
        0 <= r < ROWS
        and 0 <= c < COLS
        and state not in WALLS
    )


def step_environment(state, action):
    dr, dc = ACTIONS[action]
    candidate = (state[0] + dr, state[1] + dc)

    if not valid_state(candidate):
        candidate = state

    if candidate == GOAL:
        return candidate, 10.0, True
    if candidate == PIT:
        return candidate, -10.0, True

    return candidate, -0.1, False


def choose_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, len(ACTIONS)))

    # Random tie-breaking among equally valued actions.
    values = q_table[state[0], state[1]]
    best_actions = np.flatnonzero(
        np.isclose(values, np.max(values))
    )
    return int(rng.choice(best_actions))


def train_q_learning(epsilon, seed):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((ROWS, COLS, len(ACTIONS)), dtype=float)

    episode_returns = []
    episode_lengths = []

    for episode in range(EPISODES):
        state = START
        total_reward = 0.0
        steps = 0

        for _ in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, done = step_environment(state, action)

            old_value = q_table[state[0], state[1], action]

            if done:
                target = reward
            else:
                target = reward + GAMMA * np.max(
                    q_table[next_state[0], next_state[1]]
                )

            q_table[state[0], state[1], action] = (
                old_value + ALPHA * (target - old_value)
            )

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    return q_table, np.array(episode_returns), np.array(episode_lengths)


def evaluate_greedy_policy(q_table, episodes=100):
    """Evaluate the learned greedy policy without exploration."""
    rng = np.random.default_rng(SEED + 999)
    returns = []
    successes = 0

    for _ in range(episodes):
        state = START
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            values = q_table[state[0], state[1]]
            best_actions = np.flatnonzero(
                np.isclose(values, np.max(values))
            )
            action = int(rng.choice(best_actions))

            state, reward, done = step_environment(state, action)
            total_reward += reward

            if done:
                if state == GOAL:
                    successes += 1
                break

        returns.append(total_reward)

    return float(np.mean(returns)), successes / episodes


# ------------------------------------------------------------
# PART A: TRAIN THREE AGENTS
# ------------------------------------------------------------

all_results = {}
summary_rows = []

for index, epsilon in enumerate(EPSILONS):
    q_table, returns, lengths = train_q_learning(
        epsilon, SEED + index
    )

    all_results[epsilon] = {
        "q_table": q_table,
        "returns": returns,
        "lengths": lengths
    }

    last_100_return = float(np.mean(returns[-100:]))
    last_100_length = float(np.mean(lengths[-100:]))
    greedy_return, greedy_success = evaluate_greedy_policy(q_table)

    summary_rows.append({
        "epsilon": epsilon,
        "episodes": EPISODES,
        "mean_return_last_100": last_100_return,
        "mean_steps_last_100": last_100_length,
        "greedy_evaluation_mean_return": greedy_return,
        "greedy_evaluation_success_rate": greedy_success
    })


# ------------------------------------------------------------
# PART B: SAVE LEARNING CURVES
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

for epsilon in EPSILONS:
    returns = all_results[epsilon]["returns"]

    # Moving average over 25 episodes.
    window = 25
    moving_average = np.convolve(
        returns, np.ones(window) / window, mode="valid"
    )

    ax.plot(
        np.arange(window, EPISODES + 1),
        moving_average,
        label=f"Epsilon = {epsilon}"
    )

ax.set_xlabel("Episode")
ax.set_ylabel("Mean return (25-episode moving average)")
ax.set_title("Q-Learning: Learning Curves for Different Exploration Rates")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig1_learning_curves.png"),
    dpi=200
)
plt.close(fig)


# ------------------------------------------------------------
# PART C: SAVE GREEDY POLICY FOR EPSILON = 0.2
# ------------------------------------------------------------

chosen_epsilon = 0.2
q_table = all_results[chosen_epsilon]["q_table"]

policy_grid = np.full((ROWS, COLS), "", dtype=object)
policy_symbols = ["↑", "↓", "←", "→"]

for r in range(ROWS):
    for c in range(COLS):
        state = (r, c)

        if state in WALLS:
            policy_grid[r, c] = "W"
        elif state == START:
            policy_grid[r, c] = "S"
        elif state == GOAL:
            policy_grid[r, c] = "G"
        elif state == PIT:
            policy_grid[r, c] = "P"
        else:
            best_action = int(np.argmax(q_table[r, c]))
            policy_grid[r, c] = policy_symbols[best_action]

fig, ax = plt.subplots(figsize=(9, 5))
display_grid = np.zeros((ROWS, COLS))

for r in range(ROWS):
    for c in range(COLS):
        if (r, c) in WALLS:
            display_grid[r, c] = 1
        elif (r, c) == GOAL:
            display_grid[r, c] = 2
        elif (r, c) == PIT:
            display_grid[r, c] = 3
        elif (r, c) == START:
            display_grid[r, c] = 4

cmap = ListedColormap([
    "white", "dimgray", "lightgreen", "salmon", "lightblue"
])
ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=4)

for r in range(ROWS):
    for c in range(COLS):
        ax.text(
            c, r, policy_grid[r, c],
            ha="center", va="center",
            fontsize=16,
            color="black" if (r, c) not in WALLS else "white"
        )

ax.set_xticks(range(COLS))
ax.set_yticks(range(ROWS))
ax.set_xticklabels(range(COLS))
ax.set_yticklabels(range(ROWS))
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.set_title("Learned Greedy Policy (Epsilon = 0.2)")
ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
ax.grid(which="minor", color="black", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
fig.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig2_greedy_policy.png"),
    dpi=200
)
plt.close(fig)


# ------------------------------------------------------------
# PART D: SAVE Q-LEARNING SUMMARY TABLE
# ------------------------------------------------------------

summary_path = os.path.join(TABLE_DIR, "q_learning_results.csv")

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary_rows)

episode_path = os.path.join(TABLE_DIR, "episode_returns.csv")

with open(episode_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode"] + [
        f"epsilon_{epsilon}" for epsilon in EPSILONS
    ])

    for episode in range(EPISODES):
        writer.writerow(
            [episode + 1] + [
                all_results[epsilon]["returns"][episode]
                for epsilon in EPSILONS
            ]
        )

policy_path = os.path.join(TABLE_DIR, "greedy_policy.csv")
with open(policy_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["row"] + list(range(COLS)))
    for r in range(ROWS):
        writer.writerow([r] + list(policy_grid[r]))


# ------------------------------------------------------------
# PART E: EXECUTION REPORT
# ------------------------------------------------------------

print("=" * 68)
print("EXPERIMENT 5: REINFORCEMENT LEARNING USING Q-LEARNING")
print("=" * 68)

print("\nPART A: ENVIRONMENT CONFIGURATION")
print(f"Grid dimensions          : {ROWS} x {COLS}")
print(f"Start state              : {START}")
print(f"Goal state               : {GOAL}")
print(f"Pit state                : {PIT}")
print(f"Number of walls          : {len(WALLS)}")
print("Goal reward              : +10")
print("Pit reward               : -10")
print("Step reward              : -0.1")

print("\nPART B: LEARNING PARAMETERS")
print(f"Learning rate (alpha)    : {ALPHA}")
print(f"Discount factor (gamma)  : {GAMMA}")
print(f"Training episodes        : {EPISODES}")
print(f"Maximum steps per episode: {MAX_STEPS}")
print(f"Exploration rates        : {EPSILONS}")
print(f"Random seed              : {SEED}")

print("\nPART C: TRAINING RESULTS")
print("-" * 90)
print(
    "Epsilon | Mean return (last 100) | Mean steps | "
    "Greedy return | Greedy success"
)
print("-" * 90)

for result in summary_rows:
    print(
        f"{result['epsilon']:>7.2f} | "
        f"{result['mean_return_last_100']:>21.4f} | "
        f"{result['mean_steps_last_100']:>10.2f} | "
        f"{result['greedy_evaluation_mean_return']:>13.4f} | "
        f"{result['greedy_evaluation_success_rate'] * 100:>12.2f}%"
    )

print("\nPART D: INTERPRETATION")
print("Epsilon controls the probability of selecting a random action.")
print("A larger epsilon encourages exploration; a smaller epsilon")
print("favours exploitation of the current Q-values.")
print("The greedy evaluation measures the learned policy without")
print("epsilon-based random exploration.")

print("\nGENERATED FIGURES")
print(" - fig1_learning_curves.png")
print(" - fig2_greedy_policy.png")

print("\nGENERATED TABLES")
print(" - q_learning_results.csv")
print(" - episode_returns.csv")
print(" - greedy_policy.csv")

print(f"\nOUTPUT DIRECTORY: {OUT_DIR}")
print("\nEXPERIMENT 5 EXECUTION COMPLETED")
print("=" * 68)
