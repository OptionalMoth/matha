import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root as scipy_root  # Renamed to avoid conflict


# Define the functions
def f1(x):
    return x ** 2 - x - 1


def f2(x):
    return x ** 3 - x ** 2 - 2 * x + 1


# Derivatives for Newton-Raphson method
def df1(x):
    return 2 * x - 1


def df2(x):
    return 3 * x ** 2 - 2 * x - 2


# Bisection method implementation
def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Find root of function f using the bisection method.

    Parameters:
    f: function - The function to find roots for
    a, b: float - The interval to search for roots
    tol: float - Tolerance (default: 1e-6)
    max_iter: int - Maximum number of iterations (default: 100)

    Returns:
    float - The root found
    int - Number of iterations performed
    """
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at interval endpoints")

    iter_count = 0
    while (b - a) > tol and iter_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, iter_count
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1

    return (a + b) / 2, iter_count


# Newton-Raphson method implementation
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Find root of function f using the Newton-Raphson method.

    Parameters:
    f: function - The function to find roots for
    df: function - The derivative of f
    x0: float - Initial guess
    tol: float - Tolerance (default: 1e-6)
    max_iter: int - Maximum number of iterations (default: 100)

    Returns:
    float - The root found
    int - Number of iterations performed
    """
    x = x0
    iter_count = 0

    while iter_count < max_iter:
        fx = f(x)
        if abs(fx) < tol:
            return x, iter_count

        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero, cannot continue")

        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, iter_count

        x = x_new
        iter_count += 1

    return x, iter_count


# First, let's check the function values at various points to find proper intervals
print("Function values for f2:")
for x in np.linspace(-2, 2, 21):
    print(f"f2({x:.1f}) = {f2(x):.6f}")

# 1. Plot the functions to identify potential root locations
x = np.linspace(-3, 3, 1000)
y1 = f1(x)
y2 = f2(x)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x, y1)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.title("f(x) = x² - x - 1")
plt.xlabel("x")
plt.ylabel("f(x)")

plt.subplot(1, 2, 2)
plt.plot(x, y2)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.title("f(x) = x³ - x² - 2x + 1")
plt.xlabel("x")
plt.ylabel("f(x)")

# Add more detailed plot for f2 around potential roots
plt.figure(figsize=(10, 6))
x_detailed = np.linspace(-2, 2, 1000)
y2_detailed = f2(x_detailed)
plt.plot(x_detailed, y2_detailed)
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.title("Detailed view: f(x) = x³ - x² - 2x + 1")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tight_layout()
plt.show()

# 2. Find roots using the Bisection method
print("\nBISECTION METHOD RESULTS:")
print("-" * 50)

# For f1(x) = x² - x - 1
print("Function 1: f(x) = x² - x - 1")
# From the plot, we can see there's a root around [1, 2]
root1_bisection, iterations = bisection(f1, 1, 2)
print(f"Root: {root1_bisection:.6f} (iterations: {iterations})")

# For f2(x) = x³ - x² - 2x + 1
print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
# After examining the plot and values, use correct intervals
# From the function values above, we can see sign changes at:
# between -1.4 and -1.2 (negative to positive)
# between 0.4 and 0.6 (positive to negative)
# between 1.8 and 2.0 (negative to positive)
root2a_bisection, iterations_a = bisection(f2, -1.4, -1.2)
print(f"Root 1: {root2a_bisection:.6f} (iterations: {iterations_a})")

root2b_bisection, iterations_b = bisection(f2, 0.4, 0.6)
print(f"Root 2: {root2b_bisection:.6f} (iterations: {iterations_b})")

root2c_bisection, iterations_c = bisection(f2, 1.8, 2.0)
print(f"Root 3: {root2c_bisection:.6f} (iterations: {iterations_c})")

# 3. Find roots using the Newton-Raphson method
print("\nNEWTON-RAPHSON METHOD RESULTS:")
print("-" * 50)

# For f1(x) = x² - x - 1
print("Function 1: f(x) = x² - x - 1")
root1_newton, iterations = newton_raphson(f1, df1, 1.5)
print(f"Root: {root1_newton:.6f} (iterations: {iterations})")

# For f2(x) = x³ - x² - 2x + 1
print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
# Try various starting points for Newton-Raphson
starting_points = [-1.5, 0.5, 1.5]
newton_roots = []

for start in starting_points:
    try:
        root_val, iterations = newton_raphson(f2, df2, start)
        print(f"Starting point {start}: Root = {root_val:.6f} (iterations: {iterations})")
        newton_roots.append(root_val)
    except ValueError as e:
        print(f"Starting point {start}: Error - {e}")

# 4. Verify with scipy.optimize.root (renamed to scipy_root)
print("\nVERIFICATION WITH SCIPY:")
print("-" * 50)

# For f1(x) = x² - x - 1
print("Function 1: f(x) = x² - x - 1")
scipy_result1 = scipy_root(f1, 1.5)
print(f"Root: {scipy_result1.x[0]:.6f} (converged: {scipy_result1.success})")

# For f2(x) = x³ - x² - 2x + 1
print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
# Try multiple starting points
starting_points = [-1.5, 0.5, 1.5]
scipy_roots = []

for start in starting_points:
    result = scipy_root(f2, start)
    print(f"Starting point {start}: Root = {result.x[0]:.6f} (converged: {result.success})")
    if result.success:
        scipy_roots.append(result.x[0])

# 5. Summary and comparison
print("\nSUMMARY AND COMPARISON:")
print("-" * 50)

print("Function 1: f(x) = x² - x - 1")
print(f"Bisection method: {root1_bisection:.6f}")
print(f"Newton-Raphson method: {root1_newton:.6f}")
print(f"SciPy result: {scipy_result1.x[0]:.6f}")

print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
print("Found roots (SciPy):")

# Group similar roots (with tolerance 1e-5)
unique_roots = []
for r in scipy_roots:
    is_new = True
    for existing in unique_roots:
        if abs(r - existing) < 1e-5:
            is_new = False
            break
    if is_new:
        unique_roots.append(r)

for i, r in enumerate(sorted(unique_roots)):
    print(f"Root {i+1}: {r:.6f} (f2(x) = {f2(r):.10f})")

# Additional verification - showing function values at found roots
print("\nFunction values at found roots:")
print(f"f1({root1_newton:.6f}) = {f1(root1_newton):.10f}")

for root_val in sorted(unique_roots):
    print(f"f2({root_val:.6f}) = {f2(root_val):.10f}")

# Perform an analytical check for cubic equation
print("\nAnalytical check for cubic equation:")
print("For a cubic equation ax³ + bx² + cx + d = 0")
print(f"Our equation is: 1x³ - 1x² - 2x + 1 = 0")
print("Using the np.roots function:")
cubic_coeffs = [1, -1, -2, 1]  # Coefficients of x³ - x² - 2x + 1
analytical_roots = np.roots(cubic_coeffs)
print("Analytical roots:")
for i, r in enumerate(analytical_roots):
    if np.isreal(r):
        r_real = np.real(r)
        print(f"Root {i+1}: {r_real:.6f} (f2(x) = {f2(r_real):.10f})")
    else:
        print(f"Root {i+1}: {r} (complex root)")