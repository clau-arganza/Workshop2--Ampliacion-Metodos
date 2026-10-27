import numpy as np
import matplotlib.pyplot as plt

def exact_solution(t):
    """Return the analytical solution."""
    return 2 - np.exp(-(t**2) / 4)


def trapezoidal_method(h):
    """Solve the ODE using the implicit trapezoidal method."""

    # Define the interval and initial condition.
    t_initial = 0.0
    t_final = 4.0
    y_initial = 1.0

    # Calculate the number of steps.
    n = int(round((t_final - t_initial) / h))

    # Create the time grid, including both endpoints.
    t = np.linspace(t_initial, t_final, n + 1)

    # Create the numerical solution array.
    y = np.zeros(n + 1)
    y[0] = y_initial

    for i in range(n):

        # The implicit trapezoidal formula is:
        #
        # y[i+1] = y[i] + (h/2) * (
        #     (t[i]/2) * (2 - y[i])
        #     + (t[i+1]/2) * (2 - y[i+1])
        # )
        #
        # For this linear ODE, solve algebraically for y[i+1].

        numerator = (
            (1 - h * t[i] / 4) * y[i]
            + (h / 2) * (t[i] + t[i + 1])
        )

        denominator = 1 + h * t[i + 1] / 4

        y[i + 1] = numerator / denominator

    return t, y


# Step sizes required in the exercise.
step_sizes = [0.1, 0.01, 0.001]

colors = ["tab:red", "tab:blue", "tab:green"]
line_styles = ["--", "-.", ":"]

# Create two plots side by side.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Draw the analytical solution using a fine grid.
t_exact = np.linspace(0, 4, 1001)

axes[0].plot(
    t_exact,
    exact_solution(t_exact),
    color="black",
    linewidth=1.5,
    label="Analytical solution"
)

# Print the numerical results.
print("Exercise 2: Implicit trapezoidal method")
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

for h, color, style in zip(
    step_sizes, colors, line_styles
):

    # Calculate the numerical solution.
    t, y = trapezoidal_method(h)

    # Evaluate the analytical solution at the same grid points.
    y_exact = exact_solution(t)

    # Calculate the absolute global error at every grid point.
    absolute_error = np.abs(y_exact - y)

    endpoint_error = absolute_error[-1]
    maximum_error = np.max(absolute_error)

    print(
        f"{h:8.3f} "
        f"{len(t) - 1:8d} "
        f"{y[-1]:18.12f} "
        f"{endpoint_error:18.6e} "
        f"{maximum_error:18.6e}"
    )

    # First plot: numerical and analytical solutions.
    axes[0].plot(
        t,
        y,
        color=color,
        linestyle=style,
        linewidth=1.8,
        label=f"Trapezoidal, h = {h}"
    )

    # Second plot: absolute errors.
    # Mask zero errors because the logarithm of zero is undefined.
    error_to_plot = np.ma.masked_less_equal(
        absolute_error, 0
    )

    axes[1].semilogy(
        t,
        error_to_plot,
        color=color,
        linewidth=1.5,
        label=f"h = {h}"
    )

# Format the first plot.
axes[0].set_title("Numerical and analytical solutions")
axes[0].set_xlabel("t")
axes[0].set_ylabel("y(t)")
axes[0].set_xlim(0, 4)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Format the second plot.
axes[1].set_title("Absolute error at the grid points")
axes[1].set_xlabel("t")
axes[1].set_ylabel("|y_exact - y_numerical|")
axes[1].set_xlim(0, 4)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()