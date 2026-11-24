import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# ============================================================
# EXPERIMENT 1: TIME/FREQUENCY ANALYSIS AND PENDULUM DYNAMICS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")

for folder in [FIG, TAB]:
    os.makedirs(folder, exist_ok=True)

print("=" * 68)
print("EXPERIMENT 1: SIGNAL ANALYSIS AND PENDULUM DYNAMICS")
print("=" * 68)

# ------------------------------------------------------------
# PART A: SIGNAL GENERATION AND FREQUENCY ANALYSIS
# ------------------------------------------------------------

fs = 500.0
duration = 2.0
N = int(fs * duration)
dt = 1.0 / fs
t = np.arange(N) * dt

rng = np.random.default_rng(42)

signal = (
    np.sin(2 * np.pi * 5 * t)
    + 0.5 * np.sin(2 * np.pi * 20 * t)
    + 0.25 * np.sin(2 * np.pi * 50 * t)
    + 0.1 * rng.normal(size=N)
)

frequencies = np.fft.rfftfreq(N, d=dt)
spectrum = 2 * np.abs(np.fft.rfft(signal)) / N

print("\nPART A: SAMPLING AND FREQUENCY ANALYSIS")
print(f"Sampling frequency       : {fs:.2f} Hz")
print(f"Sampling interval        : {dt:.6f} s")
print(f"Signal duration          : {duration:.2f} s")
print(f"Number of samples        : {N}")
print(f"Frequency resolution     : {fs/N:.4f} Hz")
print(f"Nyquist frequency        : {fs/2:.2f} Hz")

peaks, _ = find_peaks(spectrum, distance=5)
dominant = sorted(peaks, key=lambda i: spectrum[i], reverse=True)[:8]
dominant = sorted(dominant, key=lambda i: frequencies[i])

print("\nDominant spectral peaks:")
for i in dominant:
    print(f"{frequencies[i]:8.2f} Hz | Amplitude: {spectrum[i]:.4f}")

with open(os.path.join(TAB, "signal_spectrum.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frequency (Hz)", "Amplitude"])
    writer.writerows(zip(frequencies, spectrum))

plt.figure(figsize=(10, 4))
plt.plot(t, signal, linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Composite Signal in Time Domain")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_time_signal.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(frequencies, spectrum)
plt.xlim(0, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Single-Sided Amplitude Spectrum")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_frequency_spectrum.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# PART B: ALIASING
# ------------------------------------------------------------

fs_alias = 8.0
duration_alias = 2.0
ta = np.arange(int(fs_alias * duration_alias)) / fs_alias
true_frequency = 5.0

xa = np.sin(2 * np.pi * true_frequency * ta)
fa = np.fft.rfftfreq(len(xa), d=1/fs_alias)
aa = np.abs(np.fft.rfft(xa))
alias_frequency = fa[np.argmax(aa[1:]) + 1]

print("\nPART B: ALIASING")
print(f"True signal frequency    : {true_frequency:.2f} Hz")
print(f"Sampling frequency       : {fs_alias:.2f} Hz")
print(f"Nyquist frequency        : {fs_alias/2:.2f} Hz")
print(f"Observed alias frequency : {alias_frequency:.2f} Hz")

plt.figure(figsize=(9, 4))
plt.stem(ta, xa)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Aliasing: 5 Hz Signal Sampled at 8 Hz")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_aliasing.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# PART C: DAMPED PENDULUM
# theta'' + (b/m)theta' + (g/L)sin(theta) = 0
# ------------------------------------------------------------

g = 9.81
L = 1.0
m = 1.0
theta_initial = np.deg2rad(40)
omega_initial = 0.0

theoretical_period = 2 * np.pi * np.sqrt(L/g)

def pendulum(t, state, damping):
    theta, omega = state
    return [
        omega,
        -(damping/m) * omega - (g/L) * np.sin(theta)
    ]

print("\nPART C: DAMPED PENDULUM")
print(f"Pendulum length           : {L:.2f} m")
print(f"Gravitational acceleration: {g:.2f} m/s^2")
print(f"Initial angle              : 40 degrees")
print(f"Small-angle theoretical T  : {theoretical_period:.5f} s")

time_eval = np.linspace(0, 15, 2000)
damping_values = [0.0, 0.3, 2.0]
solutions = {}

plt.figure(figsize=(10, 5))

for b in damping_values:
    solution = solve_ivp(
        pendulum,
        (0, 15),
        [theta_initial, omega_initial],
        args=(b,),
        t_eval=time_eval,
        rtol=1e-9,
        atol=1e-9
    )
    solutions[b] = solution

    plt.plot(
        solution.t,
        np.rad2deg(solution.y[0]),
        label=f"Damping b = {b}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Angular displacement (degrees)")
plt.title("Pendulum Response for Different Damping Coefficients")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_pendulum_damping.png"), dpi=300)
plt.close()

plt.figure(figsize=(9, 5))

for b, solution in solutions.items():
    plt.plot(
        solution.y[0],
        solution.y[1],
        label=f"Damping b = {b}"
    )

plt.xlabel("Angular displacement (rad)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Pendulum Phase Portraits")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_phase_portraits.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# PART D: PERIOD VERSUS INITIAL AMPLITUDE
# ------------------------------------------------------------

initial_angles = [5, 40, 95, 175]
period_results = []

print("\nPART D: PERIOD VERSUS INITIAL AMPLITUDE")
print(f"{'Initial angle (deg)':>22} | {'Measured period (s)':>22}")

for angle in initial_angles:
    solution = solve_ivp(
        pendulum,
        (0, 120),
        [np.deg2rad(angle), 0.0],
        args=(0.0,),
        dense_output=True,
        max_step=0.02,
        rtol=1e-9,
        atol=1e-9
    )

    sample_time = np.linspace(0, 120, 60001)
    theta = solution.sol(sample_time)[0]

    positive_peaks, _ = find_peaks(theta, prominence=1e-5)
    positive_peaks = positive_peaks[sample_time[positive_peaks] > 1.0]

    if len(positive_peaks) >= 2:
        measured_period = (
            sample_time[positive_peaks[1]]
            - sample_time[positive_peaks[0]]
        )
    else:
        measured_period = np.nan

    period_results.append([angle, measured_period])
    print(f"{angle:22.1f} | {measured_period:22.5f}")

with open(os.path.join(TAB, "pendulum_periods.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Initial angle (degrees)", "Measured period (seconds)"])
    writer.writerows(period_results)

valid_results = [
    (angle, period)
    for angle, period in period_results
    if np.isfinite(period)
]

plt.figure(figsize=(9, 5))
plt.plot(
    [x[0] for x in valid_results],
    [x[1] for x in valid_results],
    marker="o",
    label="Numerically measured period"
)
plt.axhline(
    theoretical_period,
    linestyle="--",
    label="Small-angle theoretical period"
)
plt.xlabel("Initial angular displacement (degrees)")
plt.ylabel("Period (s)")
plt.title("Pendulum Period versus Initial Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_period_vs_amplitude.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# SAVE SUMMARY AND LIST GENERATED FILES
# ------------------------------------------------------------

with open(os.path.join(TAB, "sampling_parameters.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Parameter", "Value", "Unit"])
    writer.writerows([
        ["Sampling frequency", fs, "Hz"],
        ["Sampling interval", dt, "s"],
        ["Duration", duration, "s"],
        ["Number of samples", N, "samples"],
        ["Frequency resolution", fs/N, "Hz"],
        ["Nyquist frequency", fs/2, "Hz"],
        ["Aliasing sampling frequency", fs_alias, "Hz"],
        ["True aliasing signal frequency", true_frequency, "Hz"],
        ["Observed alias frequency", alias_frequency, "Hz"],
        ["Small-angle pendulum period", theoretical_period, "s"]
    ])

print("\nGENERATED FIGURES")
for name in sorted(os.listdir(FIG)):
    print(" -", name)

print("\nGENERATED TABLES")
for name in sorted(os.listdir(TAB)):
    print(" -", name)

print("\nOUTPUT DIRECTORY:", OUT)
print("\nEXPERIMENT 1 EXECUTION COMPLETED")
print("=" * 68)
