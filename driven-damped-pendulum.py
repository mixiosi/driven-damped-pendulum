import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import pandas as pd # Using pandas for easy saving to CSV

# --- Simulation Parameters (Keep these fixed for now) ---
g = 9.81
L = 1.0
omega0_sq = g / L
gamma = 0.5
F = 1.5
omega_d = 2.0 / 3.0

# --- Data Generation Parameters ---
num_simulations = 50       # Number of different initial conditions to run
t_start = 0.0
t_end = 100.0
num_points = 3000          # Increased points for better resolution
transient_fraction = 0.25  # Fraction of initial points to discard (e.g., 25%)
output_dir = "pendulum_data" # Directory to save the data

# --- Create Output Directory ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Define ODE Function (Same as before) ---
def driven_damped_pendulum(t, y, omega0_sq, gamma, F, omega_d):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = F * np.cos(omega_d * t) - gamma * omega - omega0_sq * np.sin(theta)
    # Ensure angle stays within a reasonable range numerically if needed,
    # although solve_ivp usually handles it. Using np.sin already does.
    return [dtheta_dt, domega_dt]

# --- Generate Data Loop ---
all_runs_data = []
print(f"Starting data generation for {num_simulations} simulations...")

for i in range(num_simulations):
    print(f"Running simulation {i+1}/{num_simulations}...")

    # Generate random initial conditions within a reasonable range
    theta_init = np.random.uniform(-np.pi, np.pi)  # Angle between -pi and pi
    omega_init = np.random.uniform(-2.0, 2.0)    # Angular velocity
    y0 = [theta_init, omega_init]

    # Time span and evaluation points
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, num_points)

    # Solve the ODE system
    sol = solve_ivp(
        fun=driven_damped_pendulum,
        t_span=t_span,
        y0=y0,
        args=(omega0_sq, gamma, F, omega_d),
        t_eval=t_eval,
        method='RK45',
        # Use 'Radau' or 'BDF' if stiffness becomes an issue, but RK45 is often fine
    )

    # Extract results
    t = sol.t
    theta = sol.y[0]
    omega = sol.y[1]

    # Discard initial transient phase
    transient_points = int(transient_fraction * num_points)
    t_stable = t[transient_points:]
    theta_stable = theta[transient_points:]
    omega_stable = omega[transient_points:]

    # Store the data for this run (e.g., as a dictionary or DataFrame)
    # Adding a 'run_id' helps keep track
    run_df = pd.DataFrame({
        'time': t_stable,
        'theta': theta_stable,
        'omega': omega_stable,
        'run_id': i
    })
    all_runs_data.append(run_df)

    # Optional: Save each run individually (can be large)
    # filename = os.path.join(output_dir, f"run_{i}.csv")
    # run_df.to_csv(filename, index=False)

print("Data generation complete.")

# --- Combine all data into a single DataFrame ---
print("Combining data...")
full_dataset = pd.concat(all_runs_data, ignore_index=True)

# --- Save the Combined Dataset ---
combined_filename = os.path.join(output_dir, "combined_pendulum_data.csv")
print(f"Saving combined dataset to {combined_filename}...")
full_dataset.to_csv(combined_filename, index=False)
print("Dataset saved.")

# --- Optional: Plot one example trajectory (post-transient) ---
if num_simulations > 0:
    plt.figure(figsize=(10, 6))
    sample_run = all_runs_data[0] # Plot the first generated run after transients
    theta_wrapped_sample = (sample_run['theta'] + np.pi) % (2 * np.pi) - np.pi
    plt.plot(theta_wrapped_sample, sample_run['omega'], lw=0.5, label=f'Run 0 (Stable)')
    plt.xlabel('Angle θ (rad) [-π, π]')
    plt.ylabel('Angular Velocity ω (rad/s)')
    plt.title('Sample Phase Space Portrait (Post-Transient)')
    plt.grid(True)
    plt.legend()
    plt.show()
