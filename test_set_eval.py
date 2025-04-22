import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import matplotlib.pyplot as plt

# --- Configuration ---
processed_data_dir = "processed_data_sincos"
model_save_dir = "trained_model_sincos"
best_model_path = os.path.join(model_save_dir, "best_lstm_model_sincos.keras")

# Multi-step forecasting config
num_forecast_steps = 200 # How many steps ahead to predict
num_test_sequences_to_plot = 3 # Number of different starting sequences to forecast from

# --- 1. Load Model and Data ---
print("Loading best model...")
model = keras.models.load_model(best_model_path)
print("Model loaded.")

print(f"Loading test data and scalers from {processed_data_dir}...")
X_test_scaled = np.load(os.path.join(processed_data_dir, "X_test_sincos.npy"))
y_test_scaled = np.load(os.path.join(processed_data_dir, "y_test_sincos.npy"))
X_scaler = joblib.load(os.path.join(processed_data_dir, "X_scaler_sincos.pkl"))
y_scaler = joblib.load(os.path.join(processed_data_dir, "y_scaler_sincos.pkl"))
print("Test data and scalers loaded.")

sequence_length = X_test_scaled.shape[1]
n_features = X_test_scaled.shape[2] # Should be 3

# --- 2. Evaluate Model on Test Set (Scaled Sin/Cos/Omega Data) ---
print("Evaluating model on scaled test data...")
test_loss_scaled, test_mae_scaled = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"Test MSE (scaled, sin/cos/omega): {test_loss_scaled:.6f}")
print(f"Test MAE (scaled, sin/cos/omega): {test_mae_scaled:.6f}")

# --- 3. Make Single-Step Predictions ---
print("Making single-step predictions on test data...")
y_pred_scaled = model.predict(X_test_scaled)
print("Predictions made.")

# --- 4. Inverse Transform Predictions and Actuals (Sin/Cos/Omega) ---
print("Inverse transforming predictions and actuals...")
y_test_actual_sincos = y_scaler.inverse_transform(y_test_scaled)
y_pred_actual_sincos = y_scaler.inverse_transform(y_pred_scaled)
print("Inverse transform complete.")

# --- 5. Reconstruct Theta and Evaluate on Original Theta/Omega Scale ---
# Extract sin, cos, omega components
sin_test_actual = y_test_actual_sincos[:, 0]
cos_test_actual = y_test_actual_sincos[:, 1]
omega_test_actual = y_test_actual_sincos[:, 2]

sin_pred_actual = y_pred_actual_sincos[:, 0]
cos_pred_actual = y_pred_actual_sincos[:, 1]
omega_pred_actual = y_pred_actual_sincos[:, 2]

# Reconstruct theta using arctan2
theta_test_actual = np.arctan2(sin_test_actual, cos_test_actual)
theta_pred_actual = np.arctan2(sin_pred_actual, cos_pred_actual)

# Evaluate errors on the reconstructed theta and predicted omega
theta_mse = np.mean((theta_pred_actual - theta_test_actual)**2)
theta_mae = np.mean(np.abs(theta_pred_actual - theta_test_actual))
omega_mse = np.mean((omega_pred_actual - omega_test_actual)**2)
omega_mae = np.mean(np.abs(omega_pred_actual - omega_test_actual))

print(f"\n--- Single-Step Evaluation on Reconstructed Theta/Omega ---")
print(f"Theta MSE: {theta_mse:.6f}")
print(f"Theta MAE: {theta_mae:.6f} (radians)")
print(f"Omega MSE: {omega_mse:.6f}")
print(f"Omega MAE: {omega_mae:.6f} (rad/s)")

# --- 6. Visualize Single-Step Predictions (Theta/Omega) ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(theta_test_actual, theta_pred_actual, alpha=0.3, s=5) # Use wrapped angle automatically from arctan2
plt.xlabel("Actual Theta (rad) [-π, π]")
plt.ylabel("Predicted Theta (rad)")
plt.title("Single-Step Actual vs. Predicted Angle (θ)")
plt.grid(True)
plt.axis('equal')
plt.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=2) # Diagonal line

plt.subplot(1, 2, 2)
plt.scatter(omega_test_actual, omega_pred_actual, alpha=0.3, s=5)
plt.xlabel("Actual Omega (rad/s)")
plt.ylabel("Predicted Omega (rad/s)")
plt.title("Single-Step Actual vs. Predicted Ang. Vel. (ω)")
plt.grid(True)
plt.axis('equal')
omega_range_max = max(np.max(np.abs(omega_test_actual)), np.max(np.abs(omega_pred_actual))) * 1.1
plt.plot([-omega_range_max, omega_range_max], [-omega_range_max, omega_range_max], 'k--', lw=2)

plt.suptitle("Single-Step Prediction Performance (Sin/Cos Model)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()


# --- 7. Multi-Step Forecasting Visualization ---
print("\n--- Performing Multi-Step Forecasting ---")

plt.figure(figsize=(15, 5 * num_test_sequences_to_plot))

# Select some random starting sequences from the test set
indices_to_plot = np.random.choice(X_test_scaled.shape[0], num_test_sequences_to_plot, replace=False)

for plot_num, start_index in enumerate(indices_to_plot):
    print(f"Forecasting sequence {plot_num + 1}/{num_test_sequences_to_plot}...")

    # Get the initial sequence (scaled)
    current_sequence_scaled = X_test_scaled[start_index].copy() # Shape: (sequence_length, 3)
    # We need to add a batch dimension for the model: (1, sequence_length, 3)
    current_sequence_scaled_batch = np.expand_dims(current_sequence_scaled, axis=0)

    # Store the actual and predicted sequences (in original scale)
    forecasted_states_actual = [] # List to store [theta, omega] predictions
    actual_continuation_states = [] # List to store actual [theta, omega]

    # Inverse transform the initial input sequence to get starting actuals for plotting context
    initial_input_actual_sincos = X_scaler.inverse_transform(current_sequence_scaled)
    initial_input_theta = np.arctan2(initial_input_actual_sincos[:, 0], initial_input_actual_sincos[:, 1])
    initial_input_omega = initial_input_actual_sincos[:, 2]

    # Store initial sequence points
    for k in range(sequence_length):
         forecasted_states_actual.append([initial_input_theta[k], initial_input_omega[k]])


    # Loop to predict future steps
    for i in range(num_forecast_steps):
        # Predict the next step (scaled)
        predicted_next_state_scaled = model.predict(current_sequence_scaled_batch, verbose=0)[0] # Get prediction for the single sequence

        # Inverse transform the prediction
        predicted_next_state_sincos = y_scaler.inverse_transform(predicted_next_state_scaled.reshape(1, -1))[0]
        pred_sin, pred_cos, pred_omega = predicted_next_state_sincos
        pred_theta = np.arctan2(pred_sin, pred_cos)

        # Store the predicted state (original scale)
        forecasted_states_actual.append([pred_theta, pred_omega])

        # Get the actual next state for comparison (from y_test_scaled)
        # Need to handle index carefully - ensure we don't go out of bounds
        actual_next_index = start_index + i
        if actual_next_index < y_test_scaled.shape[0]:
            actual_next_state_scaled = y_test_scaled[actual_next_index]
            actual_next_state_sincos = y_scaler.inverse_transform(actual_next_state_scaled.reshape(1, -1))[0]
            actual_sin, actual_cos, actual_omega = actual_next_state_sincos
            actual_theta = np.arctan2(actual_sin, actual_cos)
            actual_continuation_states.append([actual_theta, actual_omega])
        else:
            # If we run out of test data for comparison, append NaN or break
            actual_continuation_states.append([np.nan, np.nan])


        # Prepare the input for the *next* prediction step
        # Append the predicted step (scaled!) to the sequence and remove the oldest step
        next_input_step_scaled = predicted_next_state_scaled.reshape(1, n_features) # Shape (1, 3)
        current_sequence_scaled = np.vstack([current_sequence_scaled[1:, :], next_input_step_scaled])
        current_sequence_scaled_batch = np.expand_dims(current_sequence_scaled, axis=0) # Update batch

    # Convert lists to arrays for easier plotting
    forecasted_states_actual = np.array(forecasted_states_actual)
    actual_continuation_states = np.array(actual_continuation_states)

    # ---- Plotting for this sequence ----
    time_steps_forecast = np.arange(forecasted_states_actual.shape[0])
    time_steps_actual = np.arange(sequence_length + actual_continuation_states.shape[0]) # Time for input + continuation

    # Plot Theta vs Time
    ax1 = plt.subplot(num_test_sequences_to_plot, 2, 2*plot_num + 1)
    # Plot actual continuation (adjust time axis to start after input sequence)
    plt.plot(time_steps_actual[sequence_length:], actual_continuation_states[:, 0], 'g.-', label='Actual Theta', markersize=4)
    # Plot forecasted theta (starts from beginning of input sequence for context)
    plt.plot(time_steps_forecast, forecasted_states_actual[:, 0], 'r.:', label='Forecasted Theta', markersize=2)
    # Highlight the input sequence portion
    plt.plot(time_steps_actual[:sequence_length], forecasted_states_actual[:sequence_length, 0], 'b.-', label='Input Theta (Actual)', markersize=4)

    plt.ylabel("Angle θ (rad)")
    plt.title(f"Multi-Step Forecast (Start Index: {start_index}) - Theta")
    plt.legend()
    plt.grid(True)
    if plot_num == num_test_sequences_to_plot - 1: plt.xlabel("Time Steps")


    # Plot Phase Space (Omega vs Theta)
    ax2 = plt.subplot(num_test_sequences_to_plot, 2, 2*plot_num + 2)
    # Plot actual continuation
    plt.plot(actual_continuation_states[:, 0], actual_continuation_states[:, 1], 'g.-', label='Actual Trajectory', markersize=4)
    # Plot forecasted trajectory (starts from beginning of input sequence)
    plt.plot(forecasted_states_actual[:, 0], forecasted_states_actual[:, 1], 'r.:', label='Forecasted Trajectory', markersize=2)
     # Highlight the input sequence portion
    plt.plot(forecasted_states_actual[:sequence_length, 0], forecasted_states_actual[:sequence_length, 1], 'b.-', label='Input Trajectory (Actual)', markersize=4)

    plt.xlabel("Angle θ (rad) [-π, π]")
    plt.ylabel("Angular Velocity ω (rad/s)")
    plt.title(f"Multi-Step Forecast (Start Index: {start_index}) - Phase Space")
    plt.legend()
    plt.grid(True)


plt.suptitle("Multi-Step Forecasting Performance (Sin/Cos Model)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for suptitle
plt.show()