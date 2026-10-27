import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    """Return the derivative dy/dt."""
    return (t / 2) * (2 - y)


def exact_solution(t):
    """Return the analytical solution."""
    return 2 - np.exp(-(t**2) / 4)


def rk4_method(h):
    """Solve the ODE using classical fourth-order Runge-Kutta."""

    n = int(round(4.0 / h))
    t = np.linspace(0.0, 4.0, n + 1)
    y = np.zeros(n + 1)

    y[0] = 1.0

    for i in range(n):

        # The k values are slopes and do not include h.

        # Slope at the beginning of the step.
        k1 = f(t[i], y[i])

        # Slope at the midpoint using k1.
        k2 = f(
            t[i] + h / 2,
            y[i] + (h / 2) * k1
        )

        # Slope at the midpoint using k2.
        k3 = f(
            t[i] + h / 2,
            y[i] + (h / 2) * k2
        )

        # Slope at the endpoint using k3.
        k4 = f(
            t[i] + h,
            y[i] + h * k3
        )

        # Combine the four slopes.
        y[i + 1] = y[i] + (h / 6) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

    return t, y


def trapezoidal_method(h):
    """Reproduce the implicit trapezoidal method from Exercise 2."""

    n = int(round(4.0 / h))
    t = np.linspace(0.0, 4.0, n + 1)
    y = np.zeros(n + 1)

    y[0] = 1.0

    for i in range(n):

        # Solve the implicit trapezoidal equation algebraically.
        numerator = (
            (1 - h * t[i] / 4) * y[i]
            + (h / 2) * (t[i] + t[i + 1])
        )

        denominator = 1 + h * t[i + 1] / 4

        y[i + 1] = numerator / denominator

    return t, y


# Required step sizes.
rk4_step_sizes = [1.0, 0.1, 0.01]
trapezoidal_step_sizes = [0.1, 0.01, 0.001]

# Store results for plotting and comparison.
rk4_results = {}
trapezoidal_results = {}

print("Exercise 4: RK4 versus implicit trapezoidal method")
print(f"Exact value at t = 4: {exact_solution(4):.12f}")
print()

print(
    f"{'Method':<14} "
    f"{'h':>7} "
    f"{'Steps':>7} "
    f"{'y(4)':>16} "
    f"{'Endpoint error':>16} "
    f"{'Maximum error':>16}"
)

print("-" * 82)

# Calculate the RK4 solutions.
for h in rk4_step_sizes:

    t, y = rk4_method(h)
    error = np.abs(exact_solution(t) - y)

    rk4_results[h] = (t, y, error)

    print(
        f"{'RK4':<14} "
        f"{h:7.3f} "
        f"{len(t) - 1:7d} "
        f"{y[-1]:16.12f} "
        f"{error[-1]:16.6e} "
        f"{np.max(error):16.6e}"
    )

# Calculate the Exercise 2 solutions using the same method.
for h in trapezoidal_step_sizes:

    t, y = trapezoidal_method(h)
    error = np.abs(exact_solution(t) - y)

    trapezoidal_results[h] = (t, y, error)

    print(
        f"{'Trapezoidal':<14} "
        f"{h:7.3f} "
        f"{len(t) - 1:7d} "
        f"{y[-1]:16.12f} "
        f"{error[-1]:16.6e} "
        f"{np.max(error):16.6e}"
    )


# ------------------------------------------------------------
# Create three plots.
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

t_exact = np.linspace(0, 4, 1001)
y_exact = exact_solution(t_exact)


# Plot 1: RK4 solutions and analytical solution.
axes[0].plot(
    t_exact,
    y_exact,
    color="black",
    linewidth=1.5,
    label="Analytical solution"
)

for h in rk4_step_sizes:

    t, y, error = rk4_results[h]

    axes[0].plot(
        t,
        y,
        linestyle="--",
        marker="o" if h == 1.0 else None,
        markersize=4,
        label=f"RK4, h = {h:g}"
    )

axes[0].set_title("RK4 solutions")
axes[0].set_xlabel("t")
axes[0].set_ylabel("y(t)")
axes[0].set_xlim(0, 4)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)


# Plot 2: compare both methods using the same step size.
# A shared step size helps isolate the effect of the method.
h_comparison = 0.1

t_rk4, y_rk4, _ = rk4_results[h_comparison]
t_trap, y_trap, _ = trapezoidal_results[h_comparison]

axes[1].plot(
    t_exact,
    y_exact,
    color="black",
    linewidth=1.5,
    label="Analytical solution"
)

axes[1].plot(
    t_rk4,
    y_rk4,
    color="tab:blue",
    linestyle="--",
    label="RK4, h = 0.1"
)

axes[1].plot(
    t_trap,
    y_trap,
    color="tab:red",
    linestyle=":",
    label="Implicit trapezoidal, h = 0.1"
)

axes[1].set_title("Comparison with Exercise 2")
axes[1].set_xlabel("t")
axes[1].set_ylabel("y(t)")
axes[1].set_xlim(0, 4)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)


# Plot 3: absolute errors for every required step size.
# Mask zero errors because log(0) is undefined.
for h in rk4_step_sizes:

    t, y, error = rk4_results[h]
    error_to_plot = np.ma.masked_less_equal(error, 0)

    axes[2].semilogy(
        t,
        error_to_plot,
        linestyle="-",
        label=f"RK4, h = {h:g}"
    )

for h in trapezoidal_step_sizes:

    t, y, error = trapezoidal_results[h]
    error_to_plot = np.ma.masked_less_equal(error, 0)

    axes[2].semilogy(
        t,
        error_to_plot,
        linestyle="--",
        label=f"Trapezoidal, h = {h:g}"
    )

axes[2].set_title("Absolute error comparison")
axes[2].set_xlabel("t")
axes[2].set_ylabel("|y_exact - y_numerical|")
axes[2].set_xlim(0, 4)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()


# ------------------------------------------------------------
# Print a numerical comparison at the shared step sizes.
# ------------------------------------------------------------

print("\nComparison at the same step size:")

for h in [0.1, 0.01]:

    rk4_max_error = np.max(rk4_results[h][2])
    trap_max_error = np.max(trapezoidal_results[h][2])

    error_ratio = trap_max_error / rk4_max_error

    print(
        f"h = {h:g}: the maximum grid error of the "
        f"trapezoidal method is {error_ratio:.2f} times "
        f"the maximum grid error of RK4."
    )

print("\nDiscussion:")
print(
    "Both methods approach the analytical solution "
    "as the step size decreases."
)
print(
    "The implicit trapezoidal method has second-order "
    "global accuracy, whereas RK4 has fourth-order "
    "global accuracy."
)
print(
    "At the shared step sizes, RK4 gives smaller errors."
)
print(
    "The solution curves almost overlap, so the error "
    "plot makes their differences easier to observe."
)
print(
    "A higher-order method is not necessarily more accurate "
    "when it uses a much larger step: RK4 with h = 1 has "
    "a larger error than the trapezoidal method with h = 0.1."
)

plt.show()