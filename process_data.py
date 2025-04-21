import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
data_path = "pendulum_data/combined_pendulum_data.csv"
sequence_length = 10  # Number of past time steps to use for prediction
test_size = 0.15      # Percentage of runs for the test set
validation_size = 0.15 # Percentage of runs for the validation set (from remaining)
output_dir = "processed_data" # Directory to save processed data

# --- Create Output Directory ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Load Data ---
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)
print("Data loaded.")
print(f"Total data points: {len(df)}")
print(f"Number of unique runs: {df['run_id'].nunique()}")

# --- 2. Create Sequences and Targets ---
# We will store features (X) and targets (y) along with their run_id
X_sequences = [] # Input sequences: list of arrays, each array is (sequence_length, 2)
y_targets = []   # Target states: list of arrays, each array is (2,)
sequence_run_ids = [] # Corresponding run_id for each sequence

print(f"Creating sequences with length {sequence_length}...")

# Process each run separately
for run_id, group in df.groupby('run_id'):
    # Sort by time just in case (should be sorted from solve_ivp t_eval)
    group = group.sort_values('time')
    theta_run = group['theta'].values
    omega_run = group['omega'].values

    # Create sequences for this run
    # A sequence starts at index i and ends at i + sequence_length - 1
    # The target is at index i + sequence_length
    # So, valid start indices for a sequence are from 0 up to len(run) - sequence_length - 1
    for i in range(len(group) - sequence_length):
        # Input features: Sequence of (theta, omega) pairs
        input_seq = np.vstack([theta_run[i : i + sequence_length],
                               omega_run[i : i + sequence_length]]).T # Transpose to get (length, 2)

        # Target: (theta, omega) at the next time step
        target = np.array([theta_run[i + sequence_length],
                           omega_run[i + sequence_length]])

        X_sequences.append(input_seq)
        y_targets.append(target)
        sequence_run_ids.append(run_id)

print(f"Created {len(X_sequences)} total sequences.")

# Convert lists to numpy arrays
X_sequences = np.array(X_sequences) # Shape: (num_sequences, sequence_length, 2)
y_targets = np.array(y_targets)     # Shape: (num_sequences, 2)
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
# X scaler needs to handle the 3D shape (num_sequences, sequence_length, n_features)
# y scaler handles the 2D shape (num_sequences, n_features)

# For X, we need to reshape it temporarily to (num_sequences * sequence_length, n_features)
# to fit the scaler, then reshape back.
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1]) # -1 infers size, shape[-1] is 2 (theta, omega)
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

# Fit X scaler ONLY on the training data reshaped
X_scaler = StandardScaler()
X_scaler.fit(X_train_reshaped)

# Apply scaler to all sets (reshaped first, then reshape back)
X_train_scaled_reshaped = X_scaler.transform(X_train_reshaped)
X_val_scaled_reshaped = X_scaler.transform(X_val_reshaped)
X_test_scaled_reshaped = X_scaler.transform(X_test_reshaped)

# Reshape back to (num_sequences, sequence_length, n_features)
X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)
X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)


# Fit y scaler ONLY on the training data (which is already 2D)
y_scaler = StandardScaler()
y_scaler.fit(y_train)

# Apply scaler to all sets
y_train_scaled = y_scaler.transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

print("\nData scaled.")


# --- 5. Save Processed Data and Scalers ---
print(f"\nSaving processed data and scalers to {output_dir}...")

np.save(os.path.join(output_dir, "X_train.npy"), X_train_scaled)
np.save(os.path.join(output_dir, "y_train.npy"), y_train_scaled)
np.save(os.path.join(output_dir, "X_val.npy"), X_val_scaled)
np.save(os.path.join(output_dir, "y_val.npy"), y_val_scaled)
np.save(os.path.join(output_dir, "X_test.npy"), X_test_scaled)
np.save(os.path.join(output_dir, "y_test.npy"), y_test_scaled)

# Save scalers (need joblib or pickle)
import joblib
joblib.dump(X_scaler, os.path.join(output_dir, "X_scaler.pkl"))
joblib.dump(y_scaler, os.path.join(output_dir, "y_scaler.pkl"))

print("Processed data and scalers saved.")
