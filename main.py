import serial
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import webbrowser
from tkinter import messagebox
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\eeg_ec\ecg_data.csv")

# Load the model
loaded_model = load_model(r'C:\Users\User\OneDrive\Desktop\eeg_ec\ecg_model.h5')

# Convert labels to numeric values
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Initialize variables
received_values = []
threshold = 20  # Number of values to collect before making a prediction

# Connect to the NodeMCU via serial
ser = serial.Serial('COM3', 9600)  # Adjust COM port as needed
ser.flush()

while True:
    if ser.in_waiting > 0:
        # Read the serial data, handling potential decoding errors
        try:
            value = float(ser.readline().decode('utf-8').rstrip())
        except (UnicodeDecodeError, ValueError) as e:
            print("Error decoding or converting to float:", e)
            continue
        
        received_values.append(value)
        
        # Once we have enough values, make a prediction
        if len(received_values) >= threshold:
            received_values = received_values[-threshold:]  # Keep only the latest 'threshold' values
            
            # Prepare the data for prediction
            X_input = np.array(received_values).reshape(-1, 1)
            
            # Predict the labels
            predictions = loaded_model.predict(X_input)
            predicted_labels = ["Stress" if pred.argmax() == 1 else "No Stress" for pred in predictions]
            
            # Count the number of stress and no stress instances
            stress_count = np.sum(predictions.argmax(axis=1) == 1)
            no_stress_count = np.sum(predictions.argmax(axis=1) == 0)
            
            # Display the result
            if stress_count > no_stress_count:
                messagebox.showinfo("Stress Relief Songs",
                                    "Here are some stress-relief songs:\n\n"
                                    "1. Stress-free song 1\n"
                                    "2. Stress-free song 2\n")
                
                # Open browser when the link is clicked
                webbrowser.open_new("https://www.youtube.com/results?search_query=stress+free+music")
            else:
                messagebox.showinfo("Prediction", f"Stress instances: {stress_count}\nNo Stress instances: {no_stress_count}\n\nRecommendation: {'Listen to music' if stress_count > no_stress_count else 'No music needed'}")
                
            # Reset received values for the next round
            received_values = []

# Close the serial connection when done
ser.close()
