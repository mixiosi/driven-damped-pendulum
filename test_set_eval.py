import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import matplotlib.pyplot as plt

# --- Configuration ---
processed_data_dir = "processed_data"
model_save_dir = "trained_model"
best_model_path = os.path.join(model_save_dir, "best_lstm_model.keras")

# --- 1. Load Model and Data ---
print("Loading best model...")
model = keras.models.load_model(best_model_path)
print("Model loaded.")

print("Loading test data and scalers...")
X_test = np.load(os.path.join(processed_data_dir, "X_test.npy"))
y_test = np.load(os.path.join(processed_data_dir, "y_test.npy"))
X_scaler = joblib.load(os.path.join(processed_data_dir, "X_scaler.pkl"))
y_scaler = joblib.load(os.path.join(processed_data_dir, "y_scaler.pkl"))
print("Test data and scalers loaded.")

# --- 2. Evaluate Model on Test Set (Scaled Data) ---
print("Evaluating model on scaled test data...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE (scaled): {test_loss:.6f}")
print(f"Test MAE (scaled): {test_mae:.6f}")

# --- 3. Make Predictions ---
print("Making predictions on test data...")
y_pred_scaled = model.predict(X_test)
print("Predictions made.")
print(f"y_pred_scaled shape: {y_pred_scaled.shape}")

# --- 4. Inverse Transform Predictions and Actuals ---
print("Inverse transforming predictions and actuals...")
y_test_actual = y_scaler.inverse_transform(y_test)
y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
print("Inverse transform complete.")

# --- 5. Evaluate on Inverse Transformed Data (Original Scale) ---
# Calculate errors in original units
mse_actual = np.mean((y_pred_actual - y_test_actual)**2)
mae_actual = np.mean(np.abs(y_pred_actual - y_test_actual))

print(f"\n--- Evaluation on Original Scale ---")
print(f"Test MSE (original scale): {mse_actual:.6f}")
print(f"Test MAE (original scale): {mae_actual:.6f}")
# Note: MAE is often more interpretable than MSE

# Separate theta and omega for easier analysis/plotting
theta_test_actual = y_test_actual[:, 0]
omega_test_actual = y_test_actual[:, 1]
theta_pred_actual = y_pred_actual[:, 0]
omega_pred_actual = y_pred_actual[:, 1]


# --- 6. Visualize Predictions vs Actuals ---

# Plotting the actual vs predicted theta and omega for a subset of test samples
# Since X_test and y_test are not sequential by default (they are just batches of sequences),
# we can pick random points or specifically reconstruct sequences for better visualization.
# For simplicity here, let's plot a sample of individual predictions.

# Option A: Scatter plot of predicted vs actual values (shows overall correlation/accuracy)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(theta_test_actual, theta_pred_actual, alpha=0.5, s=10)
plt.xlabel("Actual Theta")
plt.ylabel("Predicted Theta")
plt.title("Actual vs. Predicted Angle (θ)")
plt.grid(True)
plt.axis('equal') # Make axes equal scale
plt.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', lw=2) # Add diagonal line y=x

plt.subplot(1, 2, 2)
plt.scatter(omega_test_actual, omega_pred_actual, alpha=0.5, s=10)
plt.xlabel("Actual Omega")
plt.ylabel("Predicted Omega")
plt.title("Actual vs. Predicted Angular Velocity (ω)")
plt.grid(True)
plt.axis('equal')
# Estimate reasonable range for omega based on data or plots
omega_range = max(np.max(np.abs(omega_test_actual)), np.max(np.abs(omega_pred_actual))) * 1.1
plt.plot([-omega_range, omega_range], [-omega_range, omega_range], 'k--', lw=2)


plt.tight_layout()
plt.show()


# Option B: Plotting a short sequence prediction (more insightful for dynamics)
# This requires reconstructing a sequence from the test set and predicting step-by-step.
# Let's find the starting index of one of the test runs to plot a continuous sequence.
# We need the original dataframe or run_ids saved from step 4 to do this easily.
# For this script, let's just grab the first sequence in X_test and predict its *next* step.
# A more advanced visualization would involve predicting multiple steps ahead recursively.

# Function to grab a random sequence from the test set and its corresponding target
def get_random_test_sequence(X_test, y_test, y_scaler, X_scaler):
    idx = np.random.randint(0, X_test.shape[0])
    input_seq_scaled = X_test[idx:idx+1] # Keep batch dimension (1, sequence_length, n_features)
    target_scaled = y_test[idx]

    # Inverse transform the target
    target_actual = y_scaler.inverse_transform(target_scaled.reshape(1, -1))[0] # Reshape for scaler

    # Inverse transform the input sequence (optional, for context)
    input_seq_reshaped_scaled = input_seq_scaled.reshape(-1, input_seq_scaled.shape[-1])
    input_seq_actual = X_scaler.inverse_transform(input_seq_reshaped_scaled).reshape(input_seq_scaled.shape)

    return input_seq_scaled, input_seq_actual, target_actual # Return scaled input for model, and actuals for context/comparison


input_seq_scaled_sample, input_seq_actual_sample, target_actual_sample = get_random_test_sequence(X_test, y_test, y_scaler, X_scaler)

# Predict the next step for this sample sequence
predicted_scaled_sample = model.predict(input_seq_scaled_sample)
predicted_actual_sample = y_scaler.inverse_transform(predicted_scaled_sample)[0] # Get rid of batch dim

print("\n--- Sample Sequence Prediction ---")
print(f"Input Sequence (last step, actual): theta={input_seq_actual_sample[0, -1, 0]:.4f}, omega={input_seq_actual_sample[0, -1, 1]:.4f}")
print(f"Actual Next State:              theta={target_actual_sample[0]:.4f}, omega={target_actual_sample[1]:.4f}")
print(f"Predicted Next State:           theta={predicted_actual_sample[0]:.4f}, omega={predicted_actual_sample[1]:.4f}")

# Visualize this single step prediction on a phase space plot
plt.figure(figsize=(7, 6))
# Plot the input sequence points
plt.plot(input_seq_actual_sample[0, :, 0], input_seq_actual_sample[0, :, 1], 'o-', label='Input Sequence (Actual)', markersize=4)
# Plot the actual next point
plt.plot(target_actual_sample[0], target_actual_sample[1], 'go', markersize=7, label='Actual Next State')
# Plot the predicted next point
plt.plot(predicted_actual_sample[0], predicted_actual_sample[1], 'rx', markersize=7, label='Predicted Next State') # 'x' marker for prediction

# Wrap angle for plotting if needed (though scattered points might be fine)
# theta_wrapped = (target_actual_sample[0] + np.pi) % (2 * np.pi) - np.pi
# plt.plot(theta_wrapped, target_actual_sample[1], 'go', markersize=7, label='Actual Next State (wrapped)')
# theta_pred_wrapped = (predicted_actual_sample[0] + np.pi) % (2 * np.pi) - np.pi
# plt.plot(theta_pred_wrapped, predicted_actual_sample[1], 'rx', markersize=7, label='Predicted Next State (wrapped)')


plt.xlabel('Angle θ (rad)')
plt.ylabel('Angular Velocity ω (rad/s)')
plt.title('Single Step Prediction in Phase Space')
plt.legend()
plt.grid(True)
plt.show()


# --- Option C: Multi-step forecasting (More complex, but powerful demo) ---
# To truly see how well the model predicts dynamics, you can take a test sequence,
# predict the next step, use that prediction as part of the *next* input sequence,
# predict again, and repeat for many steps. This shows prediction drift.
# This is more involved and would require a separate loop.

# Example sketch (not full code):
# start_sequence = X_test[some_index] # Take one test sequence
# forecasted_sequence = []
# current_input = start_sequence.copy() # Start with the actual sequence

# for _ in range(forecast_steps): # Predict for N steps
#     # Predict the next state (scaled)
#     next_state_scaled = model.predict(current_input[np.newaxis, :, :]) # Add batch dim

#     # Inverse transform to actual scale
#     next_state_actual = y_scaler.inverse_transform(next_state_scaled)[0]

#     # Append to forecast (store actual or scaled, depending on need)
#     forecasted_sequence.append(next_state_actual)

#     # Prepare next input: remove the oldest step, add the newly predicted step
#     # Need to scale the predicted step before adding it to the input
#     next_state_scaled_for_input = y_scaler.transform(next_state_actual.reshape(1, -1))
#     current_input = np.vstack([current_input[0, 1:, :], next_state_scaled_for_input]) # Assuming sequence_length = current_input.shape[1]

# # Plot the original sequence, the actual continuation, and the forecasted sequence.
# This requires having access to the actual continuation of the test sequence,
# which means knowing which run_id and index the start_sequence came from.
# This is a more advanced visualization we can add later if desired.
