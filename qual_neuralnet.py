import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# Set style for better-looking plots
plt.style.use('default')

# Initialize data
trnfile = pd.read_csv('/Users/admin/Documents/VSCODE/TP_project/RAGproject/output_files/training.csv')
tstfile = pd.read_csv('/Users/admin/Documents/VSCODE/TP_project/RAGproject/output_files/testing.csv')

# Function to safely convert string representations of lists to numpy arrays
def convert_to_array(x):
    try:
        if isinstance(x, str):
            return np.array(ast.literal_eval(x), dtype=np.float32)
        return x
    except:
        return None

# Function to pad sequences to the same length
def pad_sequence(seq, max_length):
    if len(seq) > max_length:
        return seq[:max_length]
    elif len(seq) < max_length:
        return np.pad(seq, (0, max_length - len(seq)), 'constant', constant_values=0)
    return seq

# Separating Data
X_train_struc = trnfile.iloc[1:, 1:-7]
X_train_unstr = trnfile.iloc[1:, -7:]
Y_train = trnfile.iloc[1:, 0]

X_val_struc = tstfile.iloc[1:, 1:-7]
X_val_unstr = tstfile.iloc[1:, -7:]
Y_val = tstfile.iloc[1:, 0]

# Convert structured data and labels
X_train_struc = X_train_struc.to_numpy(dtype=np.float32)
X_val_struc = X_val_struc.to_numpy(dtype=np.float32)
Y_train = Y_train.to_numpy(dtype=np.float32)
Y_val = Y_val.to_numpy(dtype=np.float32)

# Convert unstructured data and handle padding
train_arrays = [convert_to_array(x) for x in X_train_unstr.values.flatten()]
val_arrays = [convert_to_array(x) for x in X_val_unstr.values.flatten()]

# Find the maximum length
max_length = max(
    max(len(arr) for arr in train_arrays if arr is not None),
    max(len(arr) for arr in val_arrays if arr is not None)
)

# Pad all sequences to max_length
X_train_unstr = np.array([pad_sequence(arr, max_length) for arr in train_arrays if arr is not None])
X_val_unstr = np.array([pad_sequence(arr, max_length) for arr in val_arrays if arr is not None])

# Repeat Y_train for each essay
Y_train_repeated = np.tile(Y_train, 7)
Y_val_repeated = np.repeat(Y_val, 7)

# Creating the models
model_unstr = keras.Sequential([
    keras.layers.InputLayer(input_shape=(max_length,)),
    keras.layers.Embedding(input_dim=5000, output_dim=128, input_length=max_length),
    keras.layers.LSTM(units=64, return_sequences=False, dropout=0.3),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model_struc = keras.Sequential([
    keras.layers.Dense(units=100, activation='relu', input_shape=[X_train_struc.shape[1]]),
    keras.layers.Dropout(rate=.3),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dropout(rate=.3),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=.001,
    restore_best_weights=True,
    monitor='val_loss'
)

# Compile models
model_struc.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model_unstr.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training
training_unstruc = model_unstr.fit(
    X_train_unstr, 
    Y_train_repeated,
    validation_data=(X_val_unstr, Y_val_repeated),
    batch_size=32,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

training_struc = model_struc.fit(
    X_train_struc, 
    Y_train,
    validation_data=(X_val_struc, Y_val),
    batch_size=32,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

# Save models
model_struc.save('model_struc_full.h5')
model_unstr.save('model_unstr_full.h5')

# Function to plot metrics
def plot_training_metrics(history, model_type):
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{model_type} Model Training Metrics', fontsize=16, y=1.05)

    # Plot Loss
    ax1.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Over Time', pad=20)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_facecolor('#f8f8f8')

    # Plot Accuracy
    ax2.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy Over Time', pad=20)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_facecolor('#f8f8f8')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Plot metrics for both models
plot_training_metrics(training_struc, 'Structured')
plot_training_metrics(training_unstruc, 'Unstructured')

# Print final results
print("\nFinal Results:")
print("\nStructured Model:")
final_struc_loss, final_struc_acc = model_struc.evaluate(X_val_struc, Y_val, verbose=0)
print(f"Loss: {final_struc_loss:.4f}")
print(f"Accuracy: {final_struc_acc:.4f}")

print("\nUnstructured Model:")
final_unstr_loss, final_unstr_acc = model_unstr.evaluate(X_val_unstr, Y_val_repeated, verbose=0)
print(f"Loss: {final_unstr_loss:.4f}")
print(f"Accuracy: {final_unstr_acc:.4f}")