import os
import csv
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# EXPERIMENT 3: BFS AND DFS ON A GRID MAZE
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")

for folder in [FIG, TAB]:
    os.makedirs(folder, exist_ok=True)

print("=" * 68)
print("EXPERIMENT 3: BREADTH-FIRST SEARCH AND DEPTH-FIRST SEARCH")
print("=" * 68)

# ------------------------------------------------------------
# PART A: GRID ENVIRONMENT
# ------------------------------------------------------------

ROWS, COLS = 23, 35
START = (1, 1)
GOAL = (21, 33)
RANDOM_SEED = 42

# 0 = free cell, 1 = obstacle
maze = np.ones((ROWS, COLS), dtype=int)

# Keep the outer boundary blocked.
# Carve a guaranteed route from start to goal.
route = []

for r in range(1, 22):
    route.append((r, 1))

for c in range(2, 34):
    route.append((21, c))

for r, c in route:
    maze[r, c] = 0

# Add deterministic random open cells until there are exactly
# 381 free cells, including the guaranteed route.
rng = random.Random(RANDOM_SEED)

available = [
    (r, c)
    for r in range(1, ROWS - 1)
    for c in range(1, COLS - 1)
    if maze[r, c] == 1
]

rng.shuffle(available)

required_open = 381
remaining = required_open - int(np.sum(maze == 0))

for r, c in available[:remaining]:
    maze[r, c] = 0

assert np.sum(maze == 0) == required_open

print("\nPART A: MAZE CONFIGURATION")
print(f"Grid dimensions : {ROWS} x {COLS}")
print(f"Total cells     : {ROWS * COLS}")
print(f"Open cells      : {np.sum(maze == 0)}")
print(f"Blocked cells   : {np.sum(maze == 1)}")
print(f"Start position  : {START}")
print(f"Goal position   : {GOAL}")
print(f"Random seed     : {RANDOM_SEED}")

# Neighbor order: Up, Down, Left, Right
MOVES = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

def get_neighbors(position):
    r, c = position
    neighbors = []

    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc

        if (
            0 <= nr < ROWS
            and 0 <= nc < COLS
            and maze[nr, nc] == 0
        ):
            neighbors.append((nr, nc))

    return neighbors


def reconstruct_path(parent, goal):
    if goal not in parent:
        return []

    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    return path[::-1]


# ------------------------------------------------------------
# PART B: BREADTH-FIRST SEARCH
# ------------------------------------------------------------

def bfs(start, goal):
    frontier = deque([start])
    visited = {start}
    parent = {start: None}

    expanded = 0
    frontier_sizes = [len(frontier)]

    start_time = time.perf_counter()

    while frontier:
        current = frontier.popleft()
        expanded += 1

        if current == goal:
            break

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                frontier.append(neighbor)

        frontier_sizes.append(len(frontier))

    elapsed = time.perf_counter() - start_time
    path = reconstruct_path(parent, goal)

    return {
        "path": path,
        "expanded": expanded,
        "visited": len(visited),
        "frontier_sizes": frontier_sizes,
        "max_frontier": max(frontier_sizes),
        "time": elapsed
    }


# ------------------------------------------------------------
# PART C: DEPTH-FIRST SEARCH
# ------------------------------------------------------------

def dfs(start, goal):
    frontier = [start]
    visited = {start}
    parent = {start: None}

    expanded = 0
    frontier_sizes = [len(frontier)]

    start_time = time.perf_counter()

    while frontier:
        current = frontier.pop()
        expanded += 1

        if current == goal:
            break

        # Reverse insertion preserves the stated move order
        # when the stack is popped.
        for neighbor in reversed(get_neighbors(current)):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                frontier.append(neighbor)

        frontier_sizes.append(len(frontier))

    elapsed = time.perf_counter() - start_time
    path = reconstruct_path(parent, goal)

    return {
        "path": path,
        "expanded": expanded,
        "visited": len(visited),
        "frontier_sizes": frontier_sizes,
        "max_frontier": max(frontier_sizes),
        "time": elapsed
    }


# ------------------------------------------------------------
# PART D: RUN BOTH SEARCH ALGORITHMS
# ------------------------------------------------------------

bfs_result = bfs(START, GOAL)
dfs_result = dfs(START, GOAL)

if not bfs_result["path"] or not dfs_result["path"]:
    raise RuntimeError("A search algorithm failed to find the goal.")

print("\nPART B AND C: SEARCH RESULTS")
print("-" * 65)

for name, result in [
    ("BFS", bfs_result),
    ("DFS", dfs_result)
]:
    print(f"\n{name}")
    print(f"Path found          : Yes")
    print(f"Path length (edges) : {len(result['path']) - 1}")
    print(f"Nodes expanded      : {result['expanded']}")
    print(f"Nodes visited       : {result['visited']}")
    print(f"Maximum frontier    : {result['max_frontier']}")
    print(f"Execution time      : {result['time']:.6f} seconds")

# ------------------------------------------------------------
# PART E: SAVE OBSERVATION TABLE
# ------------------------------------------------------------

with open(os.path.join(TAB, "bfs_dfs_comparison.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Algorithm",
        "Path length (edges)",
        "Nodes expanded",
        "Nodes visited",
        "Maximum frontier size",
        "Execution time (seconds)"
    ])

    for name, result in [
        ("BFS", bfs_result),
        ("DFS", dfs_result)
    ]:
        writer.writerow([
            name,
            len(result["path"]) - 1,
            result["expanded"],
            result["visited"],
            result["max_frontier"],
            result["time"]
        ])

# Save the actual maze layout.
with open(os.path.join(TAB, "maze_layout.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(maze.tolist())

# ------------------------------------------------------------
# PART F: VISUALIZE MAZE AND SEARCH PATHS
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 9))

for ax, name, result in [
    (axes[0], "Breadth-First Search (BFS)", bfs_result),
    (axes[1], "Depth-First Search (DFS)", dfs_result)
]:
    ax.imshow(maze, cmap="binary", vmin=0, vmax=1)

    path = np.array(result["path"])

    ax.plot(
        path[:, 1],
        path[:, 0],
        linewidth=2,
        color="blue" if name.startswith("Breadth") else "red",
        label="Search path"
    )

    ax.scatter(START[1], START[0], color="green",
               s=100, marker="o", label="Start", zorder=5)

    ax.scatter(GOAL[1], GOAL[0], color="red",
               s=100, marker="*", label="Goal", zorder=5)

    ax.set_title(
        f"{name}\nPath length = {len(result['path']) - 1}"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right")

plt.suptitle("BFS and DFS Path Comparison on the Same Maze")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG, "fig1_bfs_dfs_maze.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# PART G: FRONTIER SIZE DURING SEARCH
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    range(len(bfs_result["frontier_sizes"])),
    bfs_result["frontier_sizes"],
    label="BFS",
    linewidth=1.5
)

plt.plot(
    range(len(dfs_result["frontier_sizes"])),
    dfs_result["frontier_sizes"],
    label="DFS",
    linewidth=1.5
)

plt.xlabel("Node expansion step")
plt.ylabel("Frontier size")
plt.title("Frontier Size During BFS and DFS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG, "fig2_frontier_size.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# PART H: SUMMARY
# ------------------------------------------------------------

print("\nKEY COMPARISON")
print(
    "BFS shortest-path property: BFS returns a shortest path "
    "when all moves have equal cost."
)
print(
    "DFS limitation: DFS does not guarantee a shortest path."
)

print("\nGENERATED FIGURES")
for name in sorted(os.listdir(FIG)):
    print(" -", name)

print("\nGENERATED TABLES")
for name in sorted(os.listdir(TAB)):
    print(" -", name)

print("\nOUTPUT DIRECTORY:", OUT)
print("\nEXPERIMENT 3 EXECUTION COMPLETED")
print("=" * 68)
