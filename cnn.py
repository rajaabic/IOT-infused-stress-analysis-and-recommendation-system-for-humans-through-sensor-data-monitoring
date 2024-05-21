import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Load the dataset
data = pd.read_csv(r"C:\Users\RETECH-01\Downloads\eeg_ec\ecg_data.csv")

# Separate features (X) and labels (y)
X = data['ECG_Value'].values
y = data['Label'].values

# Convert labels to numeric values
le = LabelEncoder()
y = le.fit_transform(y)

# Reshape X to a 3D array (samples, time steps, features)
X = X.reshape(-1, 1)

# Convert labels to one-hot encoded vectors
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(1,)))  # Flatten layer to flatten the input
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 output classes (assuming 'Label' has two categories)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save the model
model.save('ecg_model.h5')

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy and loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save the figure
plt.tight_layout()
plt.savefig(r'C:\Users\RETECH-01\Downloads\eeg_ec\accuracy_and_loss_plot.png')
plt.show()
