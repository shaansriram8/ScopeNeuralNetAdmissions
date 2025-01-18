import tensorflow as tf
from tensorflow.python import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Initialize data
trnfile = pd.read_excel('/Users/admin/Documents/VSCODE/TP_project/RAGproject/output_files/training.xlsx')
tstfile = pd.read_excel('/Users/admin/Documents/VSCODE/TP_project/RAGproject/output_files/testing.xlsx')

# Convert dataframes to numpy arrays
X_train_struc = trnfile.iloc[:, :-4]
print(X_train_struc.head())
X_train_unstr = trnfile.iloc[:, -3:]
Y_train = trnfile.iloc[0]

X_val_struc = tstfile.iloc[:, :-4]
X_val_unstr = tstfile.iloc[:, -3]
Y_val = tstfile.iloc[0]

# Creating the model
model_unstr = keras.Sequential([
    keras.layers.Embedding(input_dim=5000, output_dim=128),
    keras.layers.LSTM(units=64, return_sequences=False, dropout=0.3),  # Fixed typo LSTMV1 -> LSTM
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
    restore_best_weights=True
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

# Convert inputs to tf.data.Dataset
train_unstr_dataset = tf.data.Dataset.from_tensor_slices((X_train_unstr, Y_train)).batch(32)
val_unstr_dataset = tf.data.Dataset.from_tensor_slices((X_val_unstr, Y_val)).batch(32)
train_struc_dataset = tf.data.Dataset.from_tensor_slices((X_train_struc, Y_train)).batch(32)
val_struc_dataset = tf.data.Dataset.from_tensor_slices((X_val_struc, Y_val)).batch(32)

# Training
training_unstruc = model_unstr.fit(
    train_unstr_dataset,
    validation_data=val_unstr_dataset,
    epochs=100,
    callbacks=[early_stopping]
)

training_struc = model_struc.fit(
    train_struc_dataset,
    validation_data=val_struc_dataset,
    epochs=100,
    callbacks=[early_stopping]
)

# Save the weights
model_struc.save_weights('model_struc_weights.h5')
model_unstr.save_weights('model_unstr_weights.h5')

# Plot the loss vs val_loss data 
plt.title('Structured - Loss on Model Training (Training vs Validation)')
plt.plot(training_struc.history['loss'], label='Training Loss')
plt.plot(training_struc.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid()
plt.show()

plt.title('Unstructured - Loss on Model Training (Training vs Validation)')
plt.plot(training_unstruc.history['loss'], label='Training Loss')
plt.plot(training_unstruc.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Plot the accuracy vs val_accuracy data
plt.title('Structured - Accuracy on Model Training (Training vs Validation)')
plt.plot(training_struc.history['accuracy'], label='Training Accuracy')
plt.plot(training_struc.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.title('Unstructured - Accuracy on Model Training (Training vs Validation)')
plt.plot(training_unstruc.history['accuracy'], label='Training Accuracy')
plt.plot(training_unstruc.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.grid()
plt.show()

# # Final Test of the model to evaluate
# tst_loss_struc, tst_accuracy_struc = model_struc.evaluate(X_test_struc, Y_test, verbose=2)
# tst_loss_unstr, tst_accuracy_unstr = model_unstr.evaluate(X_test_unstr, Y_test, verbose=2)

# # Final Test of the model to predict
# predict_struc = model_struc.predict(X_test_struc, verbose=2)
# predict_unstr = model_unstr.predict(X_test_unstr, verbose=2)

# predictions = (predict_unstr + predict_struc) / 2

# # Plot the test loss and accuracy
# plt.title('Structured - Loss and Accuracy on Model Testing (Training vs Validation)')
# plt.plot([tst_loss_struc], label="Test Loss")
# plt.plot([tst_accuracy_struc], label='Test Accuracy')
# plt.legend()
# plt.grid()
# plt.show()

# plt.title('Unstructured - Loss and Accuracy on Model Testing (Training vs Validation)')
# plt.plot([tst_loss_unstr], label="Test Loss")
# plt.plot([tst_accuracy_unstr], label='Test Accuracy')
# plt.legend()
# plt.grid()
# plt.show()
