import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


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

plt.tight_layout()
plt.show()

# 2. Find roots using the Bisection method
print("BISECTION METHOD RESULTS:")
print("-" * 50)

# For f1(x) = x² - x - 1
print("Function 1: f(x) = x² - x - 1")
# From the plot, we can see there's a root around [1, 2]
root1_bisection, iterations = bisection(f1, 1, 2)
print(f"Root: {root1_bisection:.6f} (iterations: {iterations})")

# For f2(x) = x³ - x² - 2x + 1
print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
# From the plot, we can see roots around [-1, 0] and [1, 2]
root2a_bisection, iterations_a = bisection(f2, -1, 0)
print(f"Root 1: {root2a_bisection:.6f} (iterations: {iterations_a})")

root2b_bisection, iterations_b = bisection(f2, 1, 2)
print(f"Root 2: {root2b_bisection:.6f} (iterations: {iterations_b})")

# 3. Find roots using the Newton-Raphson method
print("\nNEWTON-RAPHSON METHOD RESULTS:")
print("-" * 50)

# For f1(x) = x² - x - 1
print("Function 1: f(x) = x² - x - 1")
root1_newton, iterations = newton_raphson(f1, df1, 1.5)
print(f"Root: {root1_newton:.6f} (iterations: {iterations})")

# For f2(x) = x³ - x² - 2x + 1
print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
root2a_newton, iterations_a = newton_raphson(f2, df2, -0.5)
print(f"Root 1: {root2a_newton:.6f} (iterations: {iterations_a})")

root2b_newton, iterations_b = newton_raphson(f2, df2, 1.5)
print(f"Root 2: {root2b_newton:.6f} (iterations: {iterations_b})")

# 4. Verify with scipy.optimize.root
print("\nVERIFICATION WITH SCIPY:")
print("-" * 50)

# For f1(x) = x² - x - 1
print("Function 1: f(x) = x² - x - 1")
scipy_result1 = root(f1, 1.5)
print(f"Root: {scipy_result1.x[0]:.6f} (converged: {scipy_result1.success})")

# For f2(x) = x³ - x² - 2x + 1
print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
scipy_result2a = root(f2, -0.5)
print(f"Root 1: {scipy_result2a.x[0]:.6f} (converged: {scipy_result2a.success})")

scipy_result2b = root(f2, 1.5)
print(f"Root 2: {scipy_result2b.x[0]:.6f} (converged: {scipy_result2b.success})")

# Check for a third root by trying another initial point
scipy_result2c = root(f2, 0.5)
print(f"Root 3: {scipy_result2c.x[0]:.6f} (converged: {scipy_result2c.success})")

# 5. Summary and comparison
print("\nSUMMARY AND COMPARISON:")
print("-" * 50)

print("Function 1: f(x) = x² - x - 1")
print(f"Bisection method: {root1_bisection:.6f}")
print(f"Newton-Raphson method: {root1_newton:.6f}")
print(f"SciPy result: {scipy_result1.x[0]:.6f}")

print("\nFunction 2: f(x) = x³ - x² - 2x + 1")
print("Root 1:")
print(f"Bisection method: {root2a_bisection:.6f}")
print(f"Newton-Raphson method: {root2a_newton:.6f}")
print(f"SciPy result: {scipy_result2a.x[0]:.6f}")

print("\nRoot 2:")
print(f"Bisection method: {root2b_bisection:.6f}")
print(f"Newton-Raphson method: {root2b_newton:.6f}")
print(f"SciPy result: {scipy_result2b.x[0]:.6f}")

# Check if the third attempt from SciPy found a unique root
is_unique = all(abs(scipy_result2c.x[0] - r) > 1e-5 for r in [scipy_result2a.x[0], scipy_result2b.x[0]])
if is_unique:
    print("\nRoot 3:")
    print(f"SciPy result: {scipy_result2c.x[0]:.6f}")

# Additional verification - showing function values at found roots
print("\nFunction values at found roots:")
print(f"f1({root1_newton:.6f}) = {f1(root1_newton):.10f}")
print(f"f2({root2a_newton:.6f}) = {f2(root2a_newton):.10f}")
print(f"f2({root2b_newton:.6f}) = {f2(root2b_newton):.10f}")