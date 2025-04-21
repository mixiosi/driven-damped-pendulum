import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib # Needed for saving scalers

# --- Configuration ---
data_path = "pendulum_data/combined_pendulum_data.csv"
sequence_length = 10  # Number of past time steps to use for prediction
test_size = 0.15      # Percentage of runs for the test set
validation_size = 0.15 # Percentage of runs for the validation set (from remaining)
output_dir = "processed_data_sincos" # NEW: Directory for sin/cos data

# --- Create Output Directory ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Load Data ---
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)
print("Data loaded.")
print(f"Total data points: {len(df)}")
print(f"Number of unique runs: {df['run_id'].nunique()}")

# --- Add Sin/Cos features ---
print("Adding sin(theta) and cos(theta) features...")
df['sin_theta'] = np.sin(df['theta'])
df['cos_theta'] = np.cos(df['theta'])
print("Features added.")

# --- 2. Create Sequences and Targets ---
# Features will be [sin(theta), cos(theta), omega]
# Targets will be [sin(theta_next), cos(theta_next), omega_next]
feature_cols = ['sin_theta', 'cos_theta', 'omega'] # Define feature columns

X_sequences = [] # Input sequences: list of arrays, each array is (sequence_length, 3)
y_targets = []   # Target states: list of arrays, each array is (3,)
sequence_run_ids = [] # Corresponding run_id for each sequence

print(f"Creating sequences with length {sequence_length} using {feature_cols}...")

# Process each run separately
for run_id, group in df.groupby('run_id'):
    # Sort by time just in case
    group = group.sort_values('time')

    # Extract the required feature columns as a numpy array
    features_run = group[feature_cols].values # Shape: (num_points_in_run, 3)

    # Create sequences for this run
    for i in range(len(group) - sequence_length):
        # Input features: Sequence of [sin, cos, omega] pairs
        input_seq = features_run[i : i + sequence_length, :] # Shape: (sequence_length, 3)

        # Target: [sin, cos, omega] at the next time step
        target = features_run[i + sequence_length, :] # Shape: (3,)

        X_sequences.append(input_seq)
        y_targets.append(target)
        sequence_run_ids.append(run_id)

print(f"Created {len(X_sequences)} total sequences.")

# Convert lists to numpy arrays
X_sequences = np.array(X_sequences) # Shape: (num_sequences, sequence_length, 3)
y_targets = np.array(y_targets)     # Shape: (num_sequences, 3)
sequence_run_ids = np.array(sequence_run_ids) # Shape: (num_sequences,)

print(f"X_sequences shape: {X_sequences.shape}")
print(f"y_targets shape: {y_targets.shape}")

# --- 3. Split Data by Run ID ---
# Get unique run IDs
all_run_ids = df['run_id'].unique()

# First split: Separate out the test set run_ids
train_val_run_ids, test_run_ids = train_test_split(
    all_run_ids,
    test_size=test_size,
    random_state=42 # Use a random state for reproducibility
)

# Second split: Separate validation set run_ids from the remaining (train+val)
train_run_ids, val_run_ids = train_test_split(
    train_val_run_ids,
    test_size=validation_size / (1 - test_size), # Adjust size relative to the remaining data
    random_state=42
)

print(f"Train run IDs: {len(train_run_ids)}")
print(f"Validation run IDs: {len(val_run_ids)}")
print(f"Test run IDs: {len(test_run_ids)}")

# Filter X, y based on which run_id they belong to
X_train = X_sequences[np.isin(sequence_run_ids, train_run_ids)]
y_train = y_targets[np.isin(sequence_run_ids, train_run_ids)]

X_val = X_sequences[np.isin(sequence_run_ids, val_run_ids)]
y_val = y_targets[np.isin(sequence_run_ids, val_run_ids)]

X_test = X_sequences[np.isin(sequence_run_ids, test_run_ids)]
y_test = y_targets[np.isin(sequence_run_ids, test_run_ids)]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# --- 4. Scale Data ---
# Scale features (X) and targets (y) separately
# Now features are [sin_theta, cos_theta, omega] (3 features)
# Targets are [sin_theta_next, cos_theta_next, omega_next] (3 targets)

# For X, we need to reshape it temporarily
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1]) # Shape[-1] is now 3
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

# Fit X scaler ONLY on the training data reshaped
X_scaler = StandardScaler()
X_scaler.fit(X_train_reshaped)

# Apply scaler to all sets (reshaped first, then reshape back)
X_train_scaled_reshaped = X_scaler.transform(X_train_reshaped)
X_val_scaled_reshaped = X_scaler.transform(X_val_reshaped)
X_test_scaled_reshaped = X_scaler.transform(X_test_reshaped)

# Reshape back to (num_sequences, sequence_length, n_features=3)
X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)
X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)


# Fit y scaler ONLY on the training data (now 3 targets)
y_scaler = StandardScaler()
y_scaler.fit(y_train)

# Apply scaler to all sets
y_train_scaled = y_scaler.transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

print("\nData scaled.")


# --- 5. Save Processed Data and Scalers ---
print(f"\nSaving processed data and scalers to {output_dir}...")

np.save(os.path.join(output_dir, "X_train_sincos.npy"), X_train_scaled)
np.save(os.path.join(output_dir, "y_train_sincos.npy"), y_train_scaled)
np.save(os.path.join(output_dir, "X_val_sincos.npy"), X_val_scaled)
np.save(os.path.join(output_dir, "y_val_sincos.npy"), y_val_scaled)
np.save(os.path.join(output_dir, "X_test_sincos.npy"), X_test_scaled)
np.save(os.path.join(output_dir, "y_test_sincos.npy"), y_test_scaled)

# Save scalers (need joblib or pickle)
joblib.dump(X_scaler, os.path.join(output_dir, "X_scaler_sincos.pkl"))
joblib.dump(y_scaler, os.path.join(output_dir, "y_scaler_sincos.pkl"))

print("Processed data and scalers saved.")
