import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
import os
import matplotlib.pyplot as plt

# --- Configuration ---
processed_data_dir = "processed_data"
model_save_dir = "trained_model"
log_dir = "logs/fit" # For TensorBoard logs (optional but good practice)

# Model Hyperparameters
lstm_units = 64       # Number of units in the LSTM layer(s)
dropout_rate = 0.2    # Dropout rate for regularization
learning_rate = 0.001 # Learning rate for the optimizer
epochs = 100          # Maximum number of training epochs
batch_size = 64       # Number of samples per gradient update
patience = 10         # Patience for EarlyStopping

# --- Create Output Directories ---
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# --- 1. Load Processed Data ---
print("Loading processed data...")
X_train = np.load(os.path.join(processed_data_dir, "X_train.npy"))
y_train = np.load(os.path.join(processed_data_dir, "y_train.npy"))
X_val = np.load(os.path.join(processed_data_dir, "X_val.npy"))
y_val = np.load(os.path.join(processed_data_dir, "y_val.npy"))
X_test = np.load(os.path.join(processed_data_dir, "X_test.npy"))
y_test = np.load(os.path.join(processed_data_dir, "y_test.npy"))

# Load scalers (needed later for inverse transform, but not for training)
X_scaler = joblib.load(os.path.join(processed_data_dir, "X_scaler.pkl"))
y_scaler = joblib.load(os.path.join(processed_data_dir, "y_scaler.pkl"))
print("Data loaded.")

# Get input shape from training data
sequence_length = X_train.shape[1]
n_features = X_train.shape[2] # Should be 2 (theta, omega)
input_shape = (sequence_length, n_features)
print(f"Input shape: {input_shape}")

# --- 2. Define LSTM Model ---
print("Building LSTM model...")
model = Sequential([
    Input(shape=input_shape), # Explicit Input layer is good practice
    LSTM(lstm_units, return_sequences=False), # Only return the last output
    # Optional: Add another LSTM layer (requires return_sequences=True above)
    # LSTM(lstm_units // 2, return_sequences=False),
    Dropout(dropout_rate),
    Dense(n_features) # Output layer predicts 2 values (theta, omega)
    # Activation is linear by default, suitable for regression
])

model.summary() # Print model architecture

# --- 3. Compile the Model ---
print("Compiling model...")
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='mean_squared_error', # MSE is standard for regression
              metrics=['mean_absolute_error']) # Monitor MAE as well

# --- 4. Set up Callbacks ---
# Save the best model based on validation loss
checkpoint_path = os.path.join(model_save_dir, "best_lstm_model.keras") # Use .keras format
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss', # Monitor validation loss
    save_best_only=True, # Save only the best model
    verbose=1
)

# Stop training early if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience, # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restore weights from the epoch with the best val_loss
    verbose=1
)

# Optional: TensorBoard callback for visualizing training
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# --- 5. Train the Model ---
print("Starting model training...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint, early_stopping], # Add tensorboard_callback here if using
    verbose=1 # Show progress bar
)
print("Model training complete.")

# --- 6. Plot Training History ---
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (Loss)')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Validation Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train MAE')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Validation MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)

# --- Optional: Save the final model (if not using restore_best_weights) ---
# final_model_path = os.path.join(model_save_dir, "final_lstm_model.keras")
# model.save(final_model_path)
# print(f"Final model saved to {final_model_path}")
print(f"Best model automatically saved to {checkpoint_path} by ModelCheckpoint.")
