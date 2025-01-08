import random
import math

def objective_function(x):
    """
    Objective function to minimize. For example, f(x) = x^2.
    """
    return x ** 2

def local_search(objective_function, initial_solution, step_size, max_iterations):
    """
    Local search algorithm to find the minimum of the objective function.

    Parameters:
        objective_function: Function to minimize.
        initial_solution: Starting point of the search.
        step_size: Step size for neighbors.
        max_iterations: Maximum number of iterations.

    Returns:
        Best solution and its objective value.
    """
    current_solution = initial_solution
    current_value = objective_function(current_solution)

    for iteration in range(max_iterations):
        # Generate a neighbor by adding a random step
        neighbor = current_solution + random.uniform(-step_size, step_size)
        neighbor_value = objective_function(neighbor)

        # If the neighbor is better, move to the neighbor
        if neighbor_value < current_value:
            current_solution = neighbor
            current_value = neighbor_value

        # Optionally, print progress
        print(f"Iteration {iteration + 1}: Solution = {current_solution}, Value = {current_value}")

    return current_solution, current_value

# Parameters
initial_solution = random.uniform(-10, 10)  # Random initial solution in the range [-10, 10]
step_size = 0.1                            # Step size for neighbors
max_iterations = 100                       # Maximum number of iterations

# Run local search
best_solution, best_value = local_search(objective_function, initial_solution, step_size, max_iterations)

print("\nBest Solution:", best_solution)
print("Objective Value at Best Solution:", best_value)
