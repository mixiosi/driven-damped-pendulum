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
# Change this directory to load the sin/cos data
processed_data_dir = "processed_data_sincos"
model_save_dir = "trained_model_sincos" # NEW: Save models in a different directory
log_dir = "logs/fit_sincos" # For TensorBoard logs

# Model Hyperparameters (Keep same as before unless tuning needed)
lstm_units = 64
dropout_rate = 0.2
learning_rate = 0.001
epochs = 100
batch_size = 64
patience = 10

# --- Create Output Directories ---
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# --- 1. Load Processed Data ---
print(f"Loading processed data from {processed_data_dir}...")
# Update filenames to load the sin/cos data
X_train = np.load(os.path.join(processed_data_dir, "X_train_sincos.npy"))
y_train = np.load(os.path.join(processed_data_dir, "y_train_sincos.npy"))
X_val = np.load(os.path.join(processed_data_dir, "X_val_sincos.npy"))
y_val = np.load(os.path.join(processed_data_dir, "y_val_sincos.npy"))
X_test = np.load(os.path.join(processed_data_dir, "X_test_sincos.npy"))
y_test = np.load(os.path.join(processed_data_dir, "y_test_sincos.npy")) # Not used in training, but good to confirm load

# Load scalers (needed later for inverse transform)
X_scaler = joblib.load(os.path.join(processed_data_dir, "X_scaler_sincos.pkl"))
y_scaler = joblib.load(os.path.join(processed_data_dir, "y_scaler_sincos.pkl"))
print("Data loaded.")

# Get input shape from training data
sequence_length = X_train.shape[1]
# n_features is now 3 (sin_theta, cos_theta, omega)
n_features = X_train.shape[2]
input_shape = (sequence_length, n_features)
print(f"Input shape: {input_shape}")
print(f"Output shape (target): {y_train.shape[1]}") # Should be 3

# --- 2. Define LSTM Model ---
print("Building LSTM model...")
model = Sequential([
    Input(shape=input_shape),
    LSTM(lstm_units, return_sequences=False),
    Dropout(dropout_rate),
    # NEW: Output layer has n_features (3) units for [sin(theta), cos(theta), omega]
    Dense(n_features)
])

model.summary()

# --- 3. Compile the Model ---
print("Compiling model...")
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# --- 4. Set up Callbacks ---
# Save the best model based on validation loss in the new directory
checkpoint_path = os.path.join(model_save_dir, "best_lstm_model_sincos.keras")
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True,
    verbose=1
)

# Optional: TensorBoard callback
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# --- 5. Train the Model ---
print("Starting model training...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint, early_stopping], # Add tensorboard_callback here if using
    verbose=1
)
print("Model training complete.")

# --- 6. Plot Training History ---
def plot_history(history, title="Model Training History (Sin/Cos Data)"):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (Loss)')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Validation Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train MAE')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Validation MAE')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)

print(f"Best model automatically saved to {checkpoint_path} by ModelCheckpoint.")