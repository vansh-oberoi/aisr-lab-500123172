import os
import csv
import time
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# EXPERIMENT 2: EIGHT QUEENS - CSP AND BACKTRACKING
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")

for folder in [FIG, TAB]:
    os.makedirs(folder, exist_ok=True)

print("=" * 68)
print("EXPERIMENT 2: EIGHT QUEENS - CSP AND BACKTRACKING")
print("=" * 68)

# A solution is represented as a permutation:
# index = column, value = row.
# This automatically prevents two queens sharing a row or column.

def is_solution(board):
    n = len(board)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(board[i] - board[j]) == abs(i - j):
                return False
    return True


# ------------------------------------------------------------
# METHOD 1: GENERATE AND TEST
# ------------------------------------------------------------

def generate_and_test(n):
    solution_count = 0
    first_solution = None
    candidates_tested = 0

    start = time.perf_counter()

    for board in itertools.permutations(range(n)):
        candidates_tested += 1

        if is_solution(board):
            solution_count += 1
            if first_solution is None:
                first_solution = board

    elapsed = time.perf_counter() - start

    return solution_count, candidates_tested, first_solution, elapsed


# ------------------------------------------------------------
# METHOD 2: BACKTRACKING
# ------------------------------------------------------------

def backtracking(n):
    board = [-1] * n

    used_rows = [False] * n
    descending_diagonals = [False] * (2 * n - 1)
    ascending_diagonals = [False] * (2 * n - 1)

    solution_count = 0
    candidate_checks = 0
    first_solution = None

    def search(col):
        nonlocal solution_count, candidate_checks, first_solution

        if col == n:
            solution_count += 1

            if first_solution is None:
                first_solution = board.copy()

            return

        for row in range(n):
            candidate_checks += 1

            d1 = row - col + n - 1
            d2 = row + col

            if (
                not used_rows[row]
                and not descending_diagonals[d1]
                and not ascending_diagonals[d2]
            ):
                board[col] = row
                used_rows[row] = True
                descending_diagonals[d1] = True
                ascending_diagonals[d2] = True

                search(col + 1)

                board[col] = -1
                used_rows[row] = False
                descending_diagonals[d1] = False
                ascending_diagonals[d2] = False

    start = time.perf_counter()
    search(0)
    elapsed = time.perf_counter() - start

    return solution_count, candidate_checks, first_solution, elapsed


# ------------------------------------------------------------
# DISPLAY AND COLLECT RESULTS FOR N = 4 TO 10
# ------------------------------------------------------------

results = []

print("\nSOLUTION COUNTS AND EXECUTION TIMES")
print("-" * 90)
print(
    f"{'N':>3} | {'Solutions':>10} | {'G&T candidates':>15} | "
    f"{'G&T time (s)':>13} | {'BT checks':>12} | {'BT time (s)':>12}"
)
print("-" * 90)

first_eight_solution = None

for n in range(4, 11):
    gt_count, gt_candidates, gt_solution, gt_time = generate_and_test(n)

    bt_count, bt_checks, bt_solution, bt_time = backtracking(n)

    if gt_count != bt_count:
        raise RuntimeError(f"Solution count mismatch for N={n}")

    if n == 8:
        first_eight_solution = bt_solution

    results.append([
        n, gt_count, gt_candidates, gt_time,
        bt_checks, bt_time
    ])

    print(
        f"{n:3d} | {gt_count:10d} | {gt_candidates:15d} | "
        f"{gt_time:13.6f} | {bt_checks:12d} | {bt_time:12.6f}"
    )

print("-" * 90)

# ------------------------------------------------------------
# SAVE RESULTS TABLE
# ------------------------------------------------------------

with open(os.path.join(TAB, "queens_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "N",
        "Number of solutions",
        "Generate-test candidates",
        "Generate-test time (seconds)",
        "Backtracking candidate checks",
        "Backtracking time (seconds)"
    ])
    writer.writerows(results)

# ------------------------------------------------------------
# DISPLAY FIRST N=8 SOLUTION
# ------------------------------------------------------------

print("\nFIRST VALID SOLUTION FOR N=8")
print("Column -> Row:", first_eight_solution)

print("\nChessboard representation:")
for row in range(8):
    line = ""
    for col in range(8):
        line += " Q " if first_eight_solution[col] == row else " . "
    print(line)

# Plot chessboard
fig, ax = plt.subplots(figsize=(7, 7))

board_matrix = np.fromfunction(
    lambda r, c: (r + c) % 2,
    (8, 8),
    dtype=int
)

ax.imshow(board_matrix, cmap="binary", vmin=0, vmax=1)

for col, row in enumerate(first_eight_solution):
    ax.text(
        col, row, "Q",
        ha="center", va="center",
        fontsize=24, fontweight="bold",
        color="red"
    )

ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_xticklabels([f"C{i+1}" for i in range(8)])
ax.set_yticklabels([f"R{i+1}" for i in range(8)])
ax.set_xlabel("Columns")
ax.set_ylabel("Rows")
ax.set_title("One Valid Solution to the Eight Queens Problem")
ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
ax.grid(which="minor", color="gray", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_eight_queens_solution.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# RUNTIME COMPARISON
# ------------------------------------------------------------

ns = [r[0] for r in results]
gt_times = [r[3] for r in results]
bt_times = [r[5] for r in results]

plt.figure(figsize=(9, 5))
plt.plot(ns, gt_times, marker="o", label="Generate-and-test")
plt.plot(ns, bt_times, marker="s", label="Backtracking")
plt.yscale("log")
plt.xlabel("Number of Queens (N)")
plt.ylabel("Execution time (seconds, logarithmic scale)")
plt.title("Execution Time Comparison")
plt.xticks(ns)
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_runtime_comparison.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# SEARCH EFFORT COMPARISON
# ------------------------------------------------------------

gt_candidates = [r[2] for r in results]
bt_checks = [r[4] for r in results]

plt.figure(figsize=(9, 5))
plt.plot(ns, gt_candidates, marker="o", label="Generate-and-test candidates")
plt.plot(ns, bt_checks, marker="s", label="Backtracking candidate checks")
plt.xlabel("Number of Queens (N)")
plt.ylabel("Number of candidates/checks")
plt.title("Search Effort Comparison")
plt.xticks(ns)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_search_effort.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\nKEY OBSERVATION FOR N=8")
n8 = next(r for r in results if r[0] == 8)
print(f"Number of valid solutions       : {n8[1]}")
print(f"Generate-and-test candidates    : {n8[2]}")
print(f"Backtracking candidate checks  : {n8[4]}")
print(f"Generate-and-test runtime       : {n8[3]:.6f} seconds")
print(f"Backtracking runtime            : {n8[5]:.6f} seconds")

print("\nGENERATED FIGURES")
for name in sorted(os.listdir(FIG)):
    print(" -", name)

print("\nGENERATED TABLES")
for name in sorted(os.listdir(TAB)):
    print(" -", name)

print("\nOUTPUT DIRECTORY:", OUT)
print("\nEXPERIMENT 2 EXECUTION COMPLETED")
print("=" * 68)
