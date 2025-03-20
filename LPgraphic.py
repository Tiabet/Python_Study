import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Given constraints
max_area = 2000  # cm^2 (total cross-sectional area limit)
min_radius = 0.5  # cm (minimum tube radius)

# Objective function: maximize N * r (converted to minimization)
def objective(x):
    N, r = x
    return -N * r  # Negative for maximization

# Constraints
def constraint1(x):
    N, r = x
    return max_area - (N * np.pi * r**2)  # N * π * r^2 ≤ 2000 -> 2000 - N * π * r^2 ≥ 0

def constraint2(x):
    return x[1] - min_radius  # r ≥ 0.5

# Initial guess
x0 = [10, 1]  # Starting with 10 tubes of radius 1 cm

# Define constraints
constraints = ({'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2})

# Bounds for variables
bounds = [(1, None), (0.5, None)]  # N >= 1, r >= 0.5

# Solve optimization problem
solution = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
N_opt, r_opt = solution.x
N_opt = int(round(N_opt))  # Ensure N is an integer

# Generate mesh grid for visualization
N_vals = np.linspace(1, 3000, 300)  # Extended range for better visibility
r_vals = np.linspace(0.5, 5, 300)  # Possible values of r
N_grid, r_grid = np.meshgrid(N_vals, r_vals)

# Compute constraint regions
constraint_region = (N_grid * np.pi * r_grid**2 <= max_area) & (r_grid >= min_radius)

# Compute objective function values
objective_values = N_grid * r_grid

# Plot constraints and objective function contours
plt.figure(figsize=(10, 7))
plt.contourf(N_grid, r_grid, constraint_region, levels=1, colors=['gray'], alpha=0.3)
contour = plt.contour(N_grid, r_grid, objective_values, levels=10, cmap='viridis')
plt.colorbar(contour, label='Objective Function (N * r)')

# Plot feasible region boundary
constraint_line = plt.contour(N_grid, r_grid, N_grid * np.pi * r_grid**2, levels=[max_area], colors='r', linestyles='solid')
plt.clabel(constraint_line, fmt={max_area: 'N π r² = 2000'}, inline=True, fontsize=10, colors='red')
plt.axhline(y=0.5, color='g', linestyle='dashed', label='r ≥ 0.5')

# Mark the optimal solution
plt.scatter(N_opt, r_opt, color='red', marker='o', s=100, label='Optimal Solution')

# Labels and legend
plt.xlabel("Number of Tubes (N)")
plt.ylabel("Tube Radius (r) [cm]")
plt.title("Heat Exchanger Optimization: Feasible Region and Objective Function")
plt.legend()
plt.grid()
plt.show()

# Print optimal values
print(f"Optimal number of tubes: {N_opt}")
print(f"Optimal tube radius: {r_opt:.2f} cm")