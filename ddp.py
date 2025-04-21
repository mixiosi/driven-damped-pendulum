import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. Define the ODE System Function ---
def driven_damped_pendulum(t, y, omega0_sq, gamma, F, omega_d):
    """
    Defines the system of ODEs for the driven damped pendulum.

    Args:
        t (float): Time.
        y (list or np.array): State vector [theta, omega].
                                theta = y[0] (angle in radians)
                                omega = y[1] (angular velocity in rad/s)
        omega0_sq (float): Square of the natural frequency (g/L).
        gamma (float): Damping coefficient.
        F (float): Driving force amplitude.
        omega_d (float): Driving frequency.

    Returns:
        list: Derivatives [d(theta)/dt, d(omega)/dt].
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = F * np.cos(omega_d * t) - gamma * omega - omega0_sq * np.sin(theta)
    return [dtheta_dt, domega_dt]

# --- 2. Set Simulation Parameters ---
# System parameters (can be varied to see different behaviors)
g = 9.81      # Acceleration due to gravity (m/s^2)
L = 1.0       # Length of the pendulum (m)
omega0_sq = g / L # Natural frequency squared (can set directly if preferred)
gamma = 0.5     # Damping coefficient (adjust for different damping levels)
F = 1.5       # Driving amplitude (key parameter for chaos) - Try values between 0 and ~2
omega_d = 2.0/3.0 # Driving frequency (often related to natural frequency)

# Initial conditions
theta_init = 0.0  # Initial angle (radians)
omega_init = 0.0  # Initial angular velocity (rad/s)
y0 = [theta_init, omega_init]

# Time span for the simulation
t_start = 0.0
t_end = 100.0     # Simulate for a good duration to see behavior evolve
num_points = 2000 # Number of points for evaluation (increase for smoother plots)
t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, num_points)

# --- 3. Solve the ODE System ---
print("Solving ODE system...")
sol = solve_ivp(
    fun=driven_damped_pendulum,
    t_span=t_span,
    y0=y0,
    args=(omega0_sq, gamma, F, omega_d),
    t_eval=t_eval,
    method='RK45' # Standard Runge-Kutta method, usually good
    # Consider dense_output=True if you need values at arbitrary times later
)
print("ODE system solved.")

# --- 4. Extract and Process Results ---
t = sol.t
theta = sol.y[0]
omega = sol.y[1]

# Wrap angle to [-pi, pi] for better phase space plotting
# This prevents the angle from growing indefinitely in the plot
theta_wrapped = (theta + np.pi) % (2 * np.pi) - np.pi

# --- 5. Plot the Results ---
print("Plotting results...")
plt.figure(figsize=(12, 10))

# Plot 1: Angle vs. Time
plt.subplot(3, 1, 1)
plt.plot(t, theta)
plt.title('Driven Damped Pendulum Simulation')
plt.ylabel('Angle θ (rad)')
plt.grid(True)

# Plot 2: Angular Velocity vs. Time
plt.subplot(3, 1, 2)
plt.plot(t, omega)
plt.ylabel('Angular Velocity ω (rad/s)')
plt.grid(True)

# Plot 3: Phase Space Plot (omega vs. wrapped theta)
plt.subplot(3, 1, 3)
plt.plot(theta_wrapped, omega, lw=0.5) # Use wrapped angle
plt.xlabel('Angle θ (rad) [-π, π]')
plt.ylabel('Angular Velocity ω (rad/s)')
plt.title('Phase Space Portrait')
plt.grid(True)

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()
print("Plotting complete.")

# Optional: Print final state or other info
print(f"Final angle: {theta[-1]:.2f} rad")
print(f"Final angular velocity: {omega[-1]:.2f} rad/s")
