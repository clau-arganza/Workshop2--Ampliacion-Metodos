import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Exercise 4
# Solve:
#     dy/dt = (t/2) * (2 - y)
#     y(0) = 1,  0 <= t <= 4
#
# Apply classical fourth-order Runge-Kutta with:
#     h = 1, 0.1 and 0.01.
#
# Compare with Exercise 2:
#     Heun's trapezoidal predictor-corrector method,
#     using h = 0.1, 0.01 and 0.001.
# ------------------------------------------------------------


def f(t, y):
    """Return the derivative dy/dt."""
    return (t / 2) * (2 - y)


def exact_solution(t):
    """Return the analytical solution satisfying y(0) = 1."""
    return 2 - np.exp(-(t**2) / 4)


def rk4_method(h):
    """Solve the ODE using classical fourth-order Runge-Kutta."""

    t_initial = 0.0
    t_final = 4.0
    y_initial = 1.0

    # Calculate the number of steps and create the arrays.
    n = int(round((t_final - t_initial) / h))
    t = np.linspace(t_initial, t_final, n + 1)
    y = np.zeros(n + 1)

    y[0] = y_initial

    for i in range(n):

        # First slope: beginning of the interval.
        k1 = f(t[i], y[i])

        # Second slope: midpoint, using k1.
        k2 = f(
            t[i] + h / 2,
            y[i] + (h / 2) * k1
        )

        # Third slope: midpoint, using k2.
        k3 = f(
            t[i] + h / 2,
            y[i] + (h / 2) * k2
        )

        # Fourth slope: endpoint, using k3.
        k4 = f(
            t[i] + h,
            y[i] + h * k3
        )

        # Update using the weighted average of the four slopes.
        # The k values are slopes and do not include h.
        y[i + 1] = y[i] + (h / 6) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

    return t, y


def trapezoidal_method(h):
    """Reproduce the Heun method used in Exercise 2."""

    t_initial = 0.0
    t_final = 4.0
    y_initial = 1.0

    n = int(round((t_final - t_initial) / h))
    t = np.linspace(t_initial, t_final, n + 1)
    y = np.zeros(n + 1)

    y[0] = y_initial

    for i in range(n):

        # Initial slope.
        slope_initial = f(t[i], y[i])

        # Euler predictor.
        y_predictor = y[i] + h * slope_initial

        # Slope at the predicted endpoint.
        slope_final = f(t[i + 1], y_predictor)

        # One trapezoidal correction.
        y[i + 1] = y[i] + (h / 2) * (
            slope_initial + slope_final
        )

    return t, y


# Step sizes required in Exercises 4 and 2.
rk4_step_sizes = [1.0, 0.1, 0.01]
heun_step_sizes = [0.1, 0.01, 0.001]

# Create a solution plot and an error comparison plot.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot the analytical solution on a fine grid.
t_exact = np.linspace(0, 4, 1001)

axes[0].plot(
    t_exact,
    exact_solution(t_exact),
    color="black",
    linewidth=2,
    label="Analytical solution"
)

print("Exercise 4: Runge-Kutta and comparison with Exercise 2")
print(f"Exact value at t = 4: {exact_solution(4):.12f}")
print()

print(
    f"{'Method':<8} "
    f"{'h':>7} "
    f"{'Steps':>7} "
    f"{'y(4)':>16} "
    f"{'Endpoint error':>16} "
    f"{'Maximum error':>16}"
)

print("-" * 76)

# ------------------------------------------------------------
# Calculate and plot the RK4 results.
# ------------------------------------------------------------

for h in rk4_step_sizes:

    t, y = rk4_method(h)

    # Compare with the analytical solution at the same points.
    absolute_error = np.abs(exact_solution(t) - y)

    print(
        f"{'RK4':<8} "
        f"{h:7.3f} "
        f"{len(t) - 1:7d} "
        f"{y[-1]:16.12f} "
        f"{absolute_error[-1]:16.6e} "
        f"{np.max(absolute_error):16.6e}"
    )

    axes[0].plot(
        t,
        y,
        linestyle="--",
        label=f"RK4, h = {h:g}"
    )

    # Exclude zero errors because log(0) is undefined.
    positive_error = absolute_error > 0

    axes[1].semilogy(
        t[positive_error],
        absolute_error[positive_error],
        linestyle="-",
        label=f"RK4, h = {h:g}"
    )

# ------------------------------------------------------------
# Recalculate Exercise 2 results for comparison.
# ------------------------------------------------------------

for h in heun_step_sizes:

    t, y = trapezoidal_method(h)

    absolute_error = np.abs(exact_solution(t) - y)

    print(
        f"{'Heun':<8} "
        f"{h:7.3f} "
        f"{len(t) - 1:7d} "
        f"{y[-1]:16.12f} "
        f"{absolute_error[-1]:16.6e} "
        f"{np.max(absolute_error):16.6e}"
    )

    axes[0].plot(
        t,
        y,
        linestyle=":",
        label=f"Heun, h = {h:g}"
    )

    positive_error = absolute_error > 0

    axes[1].semilogy(
        t[positive_error],
        absolute_error[positive_error],
        linestyle="--",
        label=f"Heun, h = {h:g}"
    )

# Format the solution plot.
axes[0].set_title("RK4, Heun and analytical solution")
axes[0].set_xlabel("t")
axes[0].set_ylabel("y(t)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Format the error comparison plot.
axes[1].set_title("Absolute error comparison")
axes[1].set_xlabel("t")
axes[1].set_ylabel("|y_exact - y_numerical|")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Print a short interpretation of the results.
print("\nDiscussion:")
print(
    "As the step size decreases, both methods approach "
    "the analytical solution."
)
print(
    "Heun's method has second-order global accuracy, "
    "whereas classical RK4 has fourth-order global accuracy."
)
print(
    "For the shared step sizes h = 0.1 and h = 0.01, "
    "RK4 produces smaller errors than Heun's method."
)
print(
    "However, RK4 with h = 1 is less accurate than Heun "
    "with h = 0.1: accuracy depends on the step size too."
)

plt.show()