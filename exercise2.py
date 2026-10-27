import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Exercise 2
# Solve:
#     dy/dt = (t/2) * (2 - y)
#     y(0) = 1,  0 <= t <= 4
#
# Method: Euler predictor followed by one trapezoidal correction
#         (Heun's method).
# ------------------------------------------------------------


def f(t, y):
    """Return the derivative dy/dt."""
    return (t / 2) * (2 - y)


def exact_solution(t):
    """Return the analytical solution satisfying y(0) = 1."""
    return 2 - np.exp(-(t**2) / 4)


def trapezoidal_method(h):
    """Solve the ODE using Heun's predictor-corrector method."""

    # Define the interval and initial condition.
    t_initial = 0.0
    t_final = 4.0
    y_initial = 1.0

    # Calculate the number of steps.
    n = int(round((t_final - t_initial) / h))

    # Create the time grid and an array for the numerical solution.
    t = np.linspace(t_initial, t_final, n + 1)
    y = np.zeros(n + 1)

    # Set the initial condition.
    y[0] = y_initial

    # Advance the solution one step at a time.
    for i in range(n):

        # Step 1: calculate the slope at the current point.
        slope_initial = f(t[i], y[i])

        # Step 2: use Euler's method to predict the next value.
        # y_predictor = y_i + h * f(t_i, y_i)
        y_predictor = y[i] + h * slope_initial

        # Step 3: calculate the slope at the predicted endpoint.
        slope_final = f(t[i + 1], y_predictor)

        # Step 4: correct the prediction using the average slope.
        # y_(i+1) = y_i + (h/2) * (slope_initial + slope_final)
        y[i + 1] = y[i] + (h / 2) * (
            slope_initial + slope_final
        )

    return t, y


# Step sizes required in the exercise.
step_sizes = [0.1, 0.01, 0.001]

# Create two plots:
# 1. Numerical solutions and analytical solution.
# 2. Absolute errors, to make the differences easier to see.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Use a fine grid to draw the analytical solution smoothly.
t_exact = np.linspace(0, 4, 1001)

axes[0].plot(
    t_exact,
    exact_solution(t_exact),
    color="black",
    linewidth=2,
    label="Analytical solution"
)

# Print a summary of the numerical results.
print("Exercise 2: Trapezoidal predictor-corrector method")
print(f"Exact value at t = 4: {exact_solution(4):.12f}")
print()

print(
    f"{'h':>8} "
    f"{'Steps':>8} "
    f"{'Numerical y(4)':>18} "
    f"{'Endpoint error':>18} "
    f"{'Maximum error':>18}"
)

print("-" * 74)

for h in step_sizes:

    # Solve the ODE for this step size.
    t, y = trapezoidal_method(h)

    # Evaluate the analytical solution at the same grid points.
    y_exact = exact_solution(t)

    # Calculate the absolute error at every grid point.
    absolute_error = np.abs(y_exact - y)

    # Calculate the endpoint and maximum errors.
    endpoint_error = absolute_error[-1]
    maximum_error = np.max(absolute_error)

    # Display the results.
    print(
        f"{h:8.3f} "
        f"{len(t) - 1:8d} "
        f"{y[-1]:18.12f} "
        f"{endpoint_error:18.6e} "
        f"{maximum_error:18.6e}"
    )

    # Plot the numerical solution.
    axes[0].plot(
        t,
        y,
        linestyle="--",
        label=f"Heun, h = {h}"
    )

    # Plot the absolute error on a logarithmic vertical scale.
    # Omit t = 0 because its error is zero.
    axes[1].semilogy(
        t[1:],
        absolute_error[1:],
        label=f"h = {h}"
    )

# Format the solution plot.
axes[0].set_title("Numerical and analytical solutions")
axes[0].set_xlabel("t")
axes[0].set_ylabel("y(t)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Format the error plot.
axes[1].set_title("Absolute error at the grid points")
axes[1].set_xlabel("t")
axes[1].set_ylabel("|y_exact - y_numerical|")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()