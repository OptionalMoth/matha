import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Function definitions
def f1(x):
    return x**2 - x - 1

def f2(x):
    return x**3 - x**2 - 2*x + 1

# Derivatives for Newton-Raphson
def df1(x):
    return 2*x - 1

def df2(x):
    return 3*x**2 - 2*x - 2

# Bisection method implementation
def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Finds a root of function f in interval [a,b] using Bisection method
    
    Parameters:
    f (function): The function to find roots of
    a (float): Left endpoint of interval
    b (float): Right endpoint of interval
    tol (float): Tolerance for stopping criterion
    max_iter (int): Maximum number of iterations
    
    Returns:
    float: Approximation of root
    int: Number of iterations performed
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Function must have opposite signs at endpoints")
    
    iterations = 0
    while (b - a) / 2 > tol and iterations < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, iterations
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iterations += 1
    
    return (a + b) / 2, iterations

# Newton-Raphson method implementation
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Finds a root of function f using Newton-Raphson method
    
    Parameters:
    f (function): The function to find roots of
    df (function): Derivative of f
    x0 (float): Initial guess
    tol (float): Tolerance for stopping criterion
    max_iter (int): Maximum number of iterations
    
    Returns:
    float: Approximation of root
    int: Number of iterations performed
    """
    x = x0
    iterations = 0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            break
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Zero derivative. No solution found.")
        x = x - fx / dfx
        iterations += 1
    
    return x, iterations

# Plotting function
def plot_function(f, title, x_range=(-5, 5), num_points=1000):
    """
    Plots the given function over the specified range
    
    Parameters:
    f (function): The function to plot
    title (str): Title for the plot
    x_range (tuple): Range of x values (min, max)
    num_points (int): Number of points to plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = f(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

# Function to find all roots using different methods
def find_all_roots(f, df, intervals, initial_guesses):
    """
    Finds all roots using Bisection, Newton-Raphson, and SciPy's root
    
    Parameters:
    f (function): The function
    df (function): Derivative of the function
    intervals (list): List of tuples (a,b) for Bisection method
    initial_guesses (list): List of initial guesses for Newton-Raphson
    
    Returns:
    dict: Dictionary containing roots from each method
    """
    results = {
        'Bisection': [],
        'Newton-Raphson': [],
        'SciPy': []
    }
    
    # Bisection method
    for a, b in intervals:
        root_bisect, _ = bisection(f, a, b)
        results['Bisection'].append(root_bisect)
    
    # Newton-Raphson method
    for x0 in initial_guesses:
        root_nr, _ = newton_raphson(f, df, x0)
        results['Newton-Raphson'].append(root_nr)
    
    # SciPy's root function
    for x0 in initial_guesses:
        sol = root(f, x0)
        if sol.success:
            results['SciPy'].append(sol.x[0])
    
    return results

# Main execution
if __name__ == "__main__":
    # Plot the functions to visualize roots
    plot_function(f1, "Function 1: $f(x) = x^2 - x - 1$")
    plot_function(f2, "Function 2: $f(x) = x^3 - x^2 - 2x + 1$")
    
    # Find roots for function 1
    print("Roots for f(x) = x^2 - x - 1:")
    results_f1 = find_all_roots(f1, df1, [(-1, 0), (1, 2)], [-0.5, 1.5])
    for method, roots in results_f1.items():
        print(f"{method}: {roots}")
    
    # Find roots for function 2
    print("\nRoots for f(x) = x^3 - x^2 - 2x + 1:")
    results_f2 = find_all_roots(f2, df2, [(-2, -1), (0, 1), (1, 2)], [-1.5, 0.5, 1.5])
    for method, roots in results_f2.items():
        print(f"{method}: {roots}")