import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import tkinter as tk
from tkinter import messagebox
import webbrowser
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv(r"C:\Users\RE_ME_2\OneDrive\Desktop\eeg_ec\ecg_data.csv")

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(data['ECG_Value'], data['Label'], test_size=0.2)

# Load the model
loaded_model = load_model(r'C:\Users\RE_ME_2\OneDrive\Desktop\eeg_ec\ecg_model.h5')

# Convert labels to numeric values
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Function to predict label and recommend music
def predict_and_recommend():
    # Take 20 random ECG values as test data from the dataframe
    test_ecg_values = X_test
    
    # Reshape the input data
    X_input = np.array(test_ecg_values).reshape(-1, 1)
    
    # Predict the labels
    predictions = loaded_model.predict(X_input)
    
    # Convert the predictions to human-readable format
    predicted_labels = ["Stress" if pred.argmax() == 1 else "No Stress" for pred in predictions]
    
    # Count the number of stress and no stress instances
    stress_count = np.sum(predictions.argmax(axis=1) == 1)
    no_stress_count =np.sum(predictions.argmax(axis=1) == 0)
    
    # Display the result
    if stress_count > no_stress_count:
        messagebox.showinfo("Stress Relief Songs",
                            "Here are some stress-relief songs:\n\n"
                            "1. https://www.bing.com/videos/riverview/relatedvideo?&q=stress+free+music&qpvt=stress+free+music&mid=E2034B60426AD11E8044E2034B60426AD11E8044&&FORM=VRDGAR\n"
                            "2. https://www.bing.com/videos/riverview/relatedvideo?q=stress%20free%20music&mid=CDC097B9D439D981CA9ACDC097B9D439D981CA9A&ajaxhist=0")
        
        # Open browser when the link is clicked
        webbrowser.open_new("https://www.bing.com/videos/riverview/relatedvideo?&q=stress+free+music&qpvt=stress+free+music&mid=E2034B60426AD11E8044E2034B60426AD11E8044&&FORM=VRDGAR")
        webbrowser.open_new("https://www.bing.com/videos/riverview/relatedvideo?q=stress%20free%20music&mid=CDC097B9D439D981CA9ACDC097B9D439D981CA9A&ajaxhist=0")
    else:
        messagebox.showinfo("Prediction", f"Stress instances: {stress_count}\nNo Stress instances: {no_stress_count}\n\nRecommendation: {'Listen to music' if stress_count > no_stress_count else 'No music needed'}")

# Function to display the distribution of stress and no stress instances
def plot_distribution():
    labels = ['Stress', 'No Stress']
    counts = [np.sum(y_train_encoded == 1), np.sum(y_train_encoded == 0)]
    
    plt.bar(labels, counts)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Distribution of Stress and No Stress Instances')
    plt.show()

# Create the Tkinter window
root = tk.Tk()
root.title("Stress Detection and Music Recommendation")

# Create a button to trigger prediction and music recommendation
predict_button = tk.Button(root, text="Predict and Recommend Music", command=predict_and_recommend)
predict_button.pack(pady=10)

# Create a button to display the distribution of stress and no stress instances
plot_button = tk.Button(root, text="Plot Distribution", command=plot_distribution)
plot_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
