import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# EXPERIMENT 4: HILL CLIMBING AND RANDOM RESTARTS
# Problem: 8-Queens
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

for folder in (FIG_DIR, TABLE_DIR):
    os.makedirs(folder, exist_ok=True)

SEED = 42
N = 8
TRIALS = 200
RESTART_SETTINGS = [1, 2, 5, 10, 20]
rng = np.random.default_rng(SEED)


def count_conflicts(board):
    """Count attacking queen pairs in a board represented by row per column."""
    conflicts = 0
    for c1 in range(N):
        for c2 in range(c1 + 1, N):
            if board[c1] == board[c2]:
                conflicts += 1
            elif abs(board[c1] - board[c2]) == abs(c1 - c2):
                conflicts += 1
    return conflicts


def hill_climb(initial_board=None, keep_trace=False):
    """Steepest-ascent search on the negative-conflict objective."""
    if initial_board is None:
        board = rng.integers(0, N, size=N)
    else:
        board = np.array(initial_board, dtype=int).copy()

    current = count_conflicts(board)
    trace = [current]

    for step in range(100):
        best_score = current
        best_moves = []

        # Examine every alternative row for every queen.
        for col in range(N):
            original_row = board[col]
            for row in range(N):
                if row == original_row:
                    continue

                board[col] = row
                score = count_conflicts(board)

                if score < best_score:
                    best_score = score
                    best_moves = [(col, row)]
                elif score == best_score and score < current:
                    best_moves.append((col, row))

            board[col] = original_row

        # No strictly improving neighbour: local optimum reached.
        if best_score >= current:
            break

        col, row = best_moves[rng.integers(len(best_moves))]
        board[col] = row
        current = best_score
        trace.append(current)

        if current == 0:
            break

    if keep_trace:
        return board.copy(), current, trace, step + 1
    return board.copy(), current, step + 1


def run_with_restarts(max_restarts):
    """One trial: restart until a solution is found or budget is exhausted."""
    total_steps = 0
    attempts = 0

    for _ in range(max_restarts):
        _, score, steps = hill_climb()
        attempts += 1
        total_steps += steps

        if score == 0:
            return True, attempts, total_steps

    return False, attempts, total_steps


# ------------------------------------------------------------
# PART A: ONE REPRESENTATIVE HILL-CLIMBING RUN
# ------------------------------------------------------------

sample_board, sample_score, sample_trace, sample_steps = hill_climb(
    keep_trace=True
)

# Find a successful board for the solution illustration.
solution_board = None
for _ in range(5000):
    candidate, score, _ = hill_climb()
    if score == 0:
        solution_board = candidate
        break

# ------------------------------------------------------------
# PART B: RANDOM RESTART EXPERIMENTS
# ------------------------------------------------------------

results = []
single_run_successes = 0
restart_success_rates = []

for restart_count in RESTART_SETTINGS:
    successes = 0
    attempts_total = 0
    steps_total = 0

    for _ in range(TRIALS):
        success, attempts, steps = run_with_restarts(restart_count)
        successes += int(success)
        attempts_total += attempts
        steps_total += steps

    rate = successes / TRIALS
    restart_success_rates.append(rate)

    if restart_count == 1:
        single_run_successes = successes

    results.append({
        "restarts_allowed": restart_count,
        "trials": TRIALS,
        "successful_trials": successes,
        "failed_trials": TRIALS - successes,
        "observed_success_rate": rate,
        "average_attempts_used": attempts_total / TRIALS,
        "average_search_steps": steps_total / TRIALS
    })

p = single_run_successes / TRIALS

for result in results:
    k = result["restarts_allowed"]
    result["theoretical_success_rate"] = 1 - (1 - p) ** k

# ------------------------------------------------------------
# PART C: SAVE TABLES
# ------------------------------------------------------------

restart_csv = os.path.join(TABLE_DIR, "restart_success_rates.csv")
with open(restart_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

run_csv = os.path.join(TABLE_DIR, "hill_climbing_run.csv")
with open(run_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "conflicts"])
    for step, conflicts in enumerate(sample_trace):
        writer.writerow([step, conflicts])

# ------------------------------------------------------------
# PART D: FIGURES
# ------------------------------------------------------------

# Figure 1: Example solution board.
if solution_board is not None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_xticks(np.arange(N + 1))
    ax.set_yticks(np.arange(N + 1))
    ax.grid(True, linewidth=1)
    ax.set_aspect("equal")

    for row in range(N):
        for col in range(N):
            if (row + col) % 2 == 0:
                ax.add_patch(plt.Rectangle(
                    (col, N - 1 - row), 1, 1,
                    color="lightgrey", zorder=0
                ))

    for col, row in enumerate(solution_board):
        ax.text(
            col + 0.5, N - 1 - row + 0.5, "Q",
            ha="center", va="center", fontsize=22, color="darkred"
        )

    ax.set_title("A Valid 8-Queens Solution")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_queens_solution.png"), dpi=200)
    plt.close(fig)

# Figure 2: Observed and theoretical success rate.
fig, ax = plt.subplots(figsize=(8, 5))
x = [r["restarts_allowed"] for r in results]
observed = [r["observed_success_rate"] * 100 for r in results]
theoretical = [r["theoretical_success_rate"] * 100 for r in results]

ax.plot(x, observed, marker="o", linewidth=2, label="Observed")
ax.plot(x, theoretical, marker="s", linestyle="--",
        linewidth=2, label="Theoretical (independent attempts)")
ax.set_xlabel("Maximum number of restarts")
ax.set_ylabel("Success rate (%)")
ax.set_title("8-Queens: Success Rate vs Random Restarts")
ax.set_xticks(RESTART_SETTINGS)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_success_vs_restarts.png"), dpi=200)
plt.close(fig)

# Figure 3: Conflicts during one representative run.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(sample_trace)), sample_trace, marker="o")
ax.set_xlabel("Improvement step")
ax.set_ylabel("Number of attacking queen pairs")
ax.set_title("Hill-Climbing Search Progress (One Run)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_conflict_progress.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------
# PART E: EXECUTION REPORT
# ------------------------------------------------------------

print("=" * 68)
print("EXPERIMENT 4: HILL CLIMBING AND RANDOM RESTARTS")
print("=" * 68)

print("\nPART A: EXPERIMENT CONFIGURATION")
print(f"Problem                  : {N}-Queens")
print(f"Board size               : {N} x {N}")
print("Objective                : Minimise attacking queen pairs")
print("Search strategy          : Steepest-ascent hill climbing")
print("Sideways moves           : Not allowed")
print(f"Random seed              : {SEED}")
print(f"Trials per setting       : {TRIALS}")

print("\nPART B: REPRESENTATIVE HILL-CLIMBING RUN")
print(f"Initial conflicts        : {sample_trace[0]}")
print(f"Final conflicts          : {sample_score}")
print(f"Improvement steps        : {len(sample_trace) - 1}")
print(f"Solution found           : {'Yes' if sample_score == 0 else 'No'}")

print("\nPART C: RANDOM RESTART RESULTS")
print("-" * 68)
print("Restarts | Successes | Success rate | Theoretical rate")
print("-" * 68)

for result in results:
    print(
        f"{result['restarts_allowed']:>8} | "
        f"{result['successful_trials']:>9} | "
        f"{result['observed_success_rate'] * 100:>10.2f}% | "
        f"{result['theoretical_success_rate'] * 100:>14.2f}%"
    )

print("\nPART D: INTERPRETATION")
print(f"Observed single-run success probability: {p * 100:.2f}%")
print("Random restarts provide additional opportunities to escape")
print("local optima by beginning from new randomly generated states.")
print("The theoretical estimate assumes independent attempts.")

print("\nGENERATED FIGURES")
for name in [
    "fig1_queens_solution.png",
    "fig2_success_vs_restarts.png",
    "fig3_conflict_progress.png"
]:
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        print(f" - {name}")

print("\nGENERATED TABLES")
print(" - restart_success_rates.csv")
print(" - hill_climbing_run.csv")

print(f"\nOUTPUT DIRECTORY: {OUT_DIR}")
print("\nEXPERIMENT 4 EXECUTION COMPLETED")
print("=" * 68)
