import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# EXPERIMENT 6: NEURAL NETWORK AND BACKPROPAGATION
# Task: Learn the XOR Boolean function
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

for folder in (FIG_DIR, TABLE_DIR):
    os.makedirs(folder, exist_ok=True)

SEED = 1
HIDDEN_UNITS = 4
EPOCHS = 20000
LEARNING_RATES = [0.05, 0.5, 2.0]

# XOR dataset
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def forward_pass(X, W1, b1, W2, b2):
    hidden_input = X @ W1 + b1
    hidden_output = sigmoid(hidden_input)

    output_input = hidden_output @ W2 + b2
    predictions = sigmoid(output_input)

    return hidden_output, predictions


def train_network(learning_rate, seed=SEED):
    rng = np.random.default_rng(seed)

    # Small random initial weights; zero biases.
    W1 = rng.normal(0, 1, size=(2, HIDDEN_UNITS))
    b1 = np.zeros((1, HIDDEN_UNITS))
    W2 = rng.normal(0, 1, size=(HIDDEN_UNITS, 1))
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(EPOCHS):
        # Forward propagation
        hidden_output, predictions = forward_pass(
            X, W1, b1, W2, b2
        )

        error = predictions - Y
        loss = np.mean(error ** 2)
        losses.append(loss)

        # Backpropagation for mean squared error.
        output_delta = (
            2.0 / Y.size
            * error
            * predictions
            * (1.0 - predictions)
        )

        dW2 = hidden_output.T @ output_delta
        db2 = np.sum(output_delta, axis=0, keepdims=True)

        hidden_delta = (
            output_delta @ W2.T
            * hidden_output
            * (1.0 - hidden_output)
        )

        dW1 = X.T @ hidden_delta
        db1 = np.sum(hidden_delta, axis=0, keepdims=True)

        # Gradient descent update
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    _, final_predictions = forward_pass(X, W1, b1, W2, b2)
    final_loss = float(np.mean((final_predictions - Y) ** 2))

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "losses": np.array(losses),
        "predictions": final_predictions,
        "final_loss": final_loss
    }


# ------------------------------------------------------------
# PART A: TRAIN NETWORKS FOR EACH LEARNING RATE
# ------------------------------------------------------------

models = {}

for lr in LEARNING_RATES:
    models[lr] = train_network(lr)

# ------------------------------------------------------------
# PART B: SAVE LOSS CURVES
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

for lr in LEARNING_RATES:
    losses = models[lr]["losses"]
    ax.plot(
        np.arange(1, EPOCHS + 1),
        losses,
        label=f"Learning rate = {lr}"
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean squared error")
ax.set_title("XOR Neural Network: Loss vs Training Epoch")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig1_loss_curves.png"),
    dpi=200
)
plt.close(fig)

# ------------------------------------------------------------
# PART C: SAVE PREDICTION COMPARISON
# ------------------------------------------------------------

prediction_path = os.path.join(TABLE_DIR, "xor_predictions.csv")

with open(prediction_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Input 1", "Input 2", "Target"]

    for lr in LEARNING_RATES:
        header.append(f"Prediction_lr_{lr}")
        header.append(f"Class_lr_{lr}")

    writer.writerow(header)

    for i in range(len(X)):
        row = [int(X[i, 0]), int(X[i, 1]), int(Y[i, 0])]

        for lr in LEARNING_RATES:
            pred = float(models[lr]["predictions"][i, 0])
            row.extend([pred, int(pred >= 0.5)])

        writer.writerow(row)

# ------------------------------------------------------------
# PART D: SAVE LOSS HISTORY
# ------------------------------------------------------------

loss_path = os.path.join(TABLE_DIR, "training_loss.csv")

with open(loss_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["epoch"] + [f"loss_lr_{lr}" for lr in LEARNING_RATES]
    )

    for epoch in range(EPOCHS):
        writer.writerow(
            [epoch + 1] + [
                models[lr]["losses"][epoch]
                for lr in LEARNING_RATES
            ]
        )

# ------------------------------------------------------------
# PART E: SAVE DECISION BOUNDARY FOR LR = 0.5
# ------------------------------------------------------------

chosen_lr = 0.5
model = models[chosen_lr]

grid_x, grid_y = np.meshgrid(
    np.linspace(-0.2, 1.2, 200),
    np.linspace(-0.2, 1.2, 200)
)

grid_points = np.column_stack([
    grid_x.ravel(),
    grid_y.ravel()
])

_, grid_predictions = forward_pass(
    grid_points,
    model["W1"], model["b1"],
    model["W2"], model["b2"]
)

decision_grid = grid_predictions.reshape(grid_x.shape)

fig, ax = plt.subplots(figsize=(7, 6))
contour = ax.contourf(
    grid_x, grid_y, decision_grid,
    levels=np.linspace(0, 1, 11),
    cmap="coolwarm",
    alpha=0.7
)

ax.contour(
    grid_x, grid_y, decision_grid,
    levels=[0.5],
    colors="black",
    linewidths=1.5
)

for i in range(len(X)):
    marker = "o" if Y[i, 0] == 1 else "s"
    ax.scatter(
        X[i, 0], X[i, 1],
        marker=marker,
        s=120,
        edgecolors="black",
        linewidths=1.2,
        label=f"Target {int(Y[i, 0])}"
    )

ax.set_xlabel("Input x1")
ax.set_ylabel("Input x2")
ax.set_title("XOR Decision Boundary (Learning Rate = 0.5)")
fig.colorbar(contour, ax=ax, label="Predicted output")
ax.legend()
fig.tight_layout()
fig.savefig(
    os.path.join(FIG_DIR, "fig2_decision_boundary.png"),
    dpi=200
)
plt.close(fig)

# ------------------------------------------------------------
# PART F: EXECUTION REPORT
# ------------------------------------------------------------

print("=" * 68)
print("EXPERIMENT 6: NEURAL NETWORK AND BACKPROPAGATION")
print("=" * 68)

print("\nPART A: DATASET AND NETWORK CONFIGURATION")
print("Task                     : XOR Boolean classification")
print(f"Training samples         : {len(X)}")
print("Input neurons             : 2")
print(f"Hidden neurons            : {HIDDEN_UNITS}")
print("Output neurons            : 1")
print("Activation function       : Sigmoid")
print("Loss function             : Mean Squared Error")
print("Optimisation              : Gradient Descent")
print(f"Training epochs           : {EPOCHS}")
print(f"Random seed               : {SEED}")

print("\nPART B: LEARNING RATE COMPARISON")
print("-" * 68)
print("Learning rate | Final MSE | Converged below 0.001")
print("-" * 68)

for lr in LEARNING_RATES:
    final_loss = models[lr]["final_loss"]
    print(
        f"{lr:>13.2f} | "
        f"{final_loss:>9.6f} | "
        f"{'Yes' if final_loss < 0.001 else 'No'}"
    )

print("\nPART C: XOR PREDICTIONS")
for lr in LEARNING_RATES:
    print(f"\nLearning rate = {lr}")
    print("Input       Target   Prediction   Class")

    for i in range(len(X)):
        pred = float(models[lr]["predictions"][i, 0])
        predicted_class = int(pred >= 0.5)

        print(
            f"{X[i].astype(int).tolist()}       "
            f"{int(Y[i, 0])}        "
            f"{pred:.6f}     "
            f"{predicted_class}"
        )

print("\nPART D: INTERPRETATION")
print("The XOR problem is not linearly separable.")
print("A hidden layer enables the network to learn a nonlinear")
print("decision boundary through backpropagation.")
print("Learning rate affects the speed and stability of convergence.")

print("\nGENERATED FIGURES")
print(" - fig1_loss_curves.png")
print(" - fig2_decision_boundary.png")

print("\nGENERATED TABLES")
print(" - xor_predictions.csv")
print(" - training_loss.csv")

print(f"\nOUTPUT DIRECTORY: {OUT_DIR}")
print("\nEXPERIMENT 6 EXECUTION COMPLETED")
print("=" * 68)
