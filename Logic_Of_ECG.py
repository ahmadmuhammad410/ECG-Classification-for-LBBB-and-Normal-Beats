import numpy as np
from scipy.signal import butter, filtfilt
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split



from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore
from tkinter import messagebox
from scipy.signal import savgol_filter  # Importing the savgol_filter for smoothing

import math
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import pandas as pd


# Function to apply Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to extract wavelet features
def extract_wavelet_features(signal, wavelet='db4', levels=5):
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])  # Add mean and std of coefficients
    return np.array(features)

# Function to clean data: remove NaN, Inf, and clip extreme values
def clean_data(data):
    data = np.where(np.isnan(data), 0, data)  # Replace NaN with 0
    data = np.where(np.isinf(data), 0, data)  # Replace Inf with 0
    data = np.clip(data, -1, 1)  # Adjust the range as needed
    return data

# Load data from files (for training only)
def load_data(filename, label, fs=1000):
    with open(filename, "r") as file:
        content = file.read()
    data = np.array([float(x) for x in content.strip().split('|') if x.strip()])

    # Clean the data
    data = clean_data(data)

    # Preprocessing: Mean Removal and Filtering
    mean_removed = data - np.mean(data)
    filtered = butter_bandpass_filter(mean_removed, 0.5, 40, fs)

    # Segment the signal
    segment_length = 500
    segments = [filtered[i:i + segment_length] for i in range(0, len(filtered), segment_length)]
    segments = [seg for seg in segments if len(seg) == segment_length]

    # Extract features and assign labels
    features = np.array([extract_wavelet_features(seg) for seg in segments])
    labels = np.full(len(features), label)
    return features, labels

# Training Phase
# Load training data
X_normal, y_normal = load_data("Normal_Train.txt", label=0)
X_pvc, y_pvc = load_data("LBBB_Train.txt", label=1)

# Combine training data
X = np.vstack((X_normal, X_pvc))
y = np.hstack((y_normal, y_pvc))

# Clean training data to prevent issues with Inf or NaN values
X = np.array([clean_data(x) for x in X])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Validate the model
y_val_pred = knn.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Prediction Phase
def predict_ecg(file_path, fs=1000):
    # Load and preprocess the user's file
    with open(file_path, "r") as file:
        content = file.read()
    data = np.array([float(x) for x in content.strip().split('|') if x.strip()])

    # Clean the data
    data = clean_data(data)

    # Preprocessing: Mean Removal and Filtering
    mean_removed = data - np.mean(data)
    filtered = butter_bandpass_filter(mean_removed, 0.5, 40, fs)

    # Segment the signal and extract features
    segment_length = 500
    segments = [filtered[i:i + segment_length] for i in range(0, len(filtered), segment_length)]
    segments = [seg for seg in segments if len(seg) == segment_length]
    features = np.array([extract_wavelet_features(seg) for seg in segments])

    # Clean features to avoid Inf or NaN issues
    features = np.array([clean_data(x) for x in features])

    # Predict using KNN
    predictions = knn.predict(features)

    # Return majority vote for the file
    majority_vote = np.round(np.mean(predictions))
    return "Normal" if majority_vote == 0 else "LBBB"

# Example Usage
# user_file = "Normal_Test.txt"  # Replace this with the actual file path
# result = predict_ecg(user_file)
# print(f"The predicted condition is:{result}")    
    
# Functions for ECG Processing

# Color Definitions
royal_blue = "#4169E1"      # Royal Blue
crimson = "#DC143C"         # Crimson
emerald_green = "#50C878"   # Emerald Green
gold = "#FFD700"            # Gold
slate_gray = "#708090"      # Slate Gray
purpel = "#9008FF"          #purpel
left_frame_color = "#6a0dad"
right_frame_color = "#F2F2F2"
button_color = "#FFBF00"  # Amber
green = "#008200"
light_blue = "#00FFF4"


accuracy = 0


def plot_Normal_ecg():
    # Simulated time axis (500 points)
    time = np.linspace(0, 1, 500)
    
    # Generate synthetic ECG signal components
    ecg_signal = (0.1 * np.sin(2 * np.pi * 5 * time) +    # P wave
                  -0.3 * np.exp(-((time - 0.25) ** 2) / 0.002) +  # Q wave
                  1.0 * np.exp(-((time - 0.3) ** 2) / 0.001) +    # R wave
                  -0.5 * np.exp(-((time - 0.35) ** 2) / 0.002) +  # S wave
                  0.3 * np.exp(-((time - 0.6) ** 2) / 0.01))      # T wave

    # # Apply a smoothing filter to the ECG signal
    # smoothed_ecg = savgol_filter(ecg_signal, window_length=31, polyorder=3)

    # Create a frame for the plot
    plot_frame = Frame(right_frame, bg="#000" )
    plot_frame.pack(side=TOP, fill=BOTH, expand=True)

    # plot.set_facecolor('#000000')

    # Create a figure for the plot
    fig = Figure(figsize=(5, 2.5), dpi=100)  # Adjust height to 2.5 (50% of 5)
    plot = fig.add_subplot(111)
    plot.plot(time, ecg_signal, label='ECG', color=gold)
    plot.set_title('Normal ECG Signal', fontsize=12)
    plot.set_xlabel('Time (s)', fontsize=10 , color=purpel)
    plot.set_ylabel('Amplitude (mV)', fontsize=10 , color=purpel)
    
    
    # Change the color of the X-axis and Y-axis lines (spines)
    plot.spines['bottom'].set_color(purpel)  # purpel color for X-axis
    plot.spines['left'].set_color(purpel)    # purpel color for Y-axis
    
    # Optionally, you can also change the color of the top and right spines
    plot.spines['top'].set_color('none')         # Hide the top spine
    plot.spines['right'].set_color('none') 
    
    
    plot.tick_params(axis='x', colors=purpel)  # purpel color for X-axis tick labels
    plot.tick_params(axis='y', colors=purpel)  # purpel color for Y-axis tick labels
 
    
    plot.grid(color=purpel) 
    
    # plot.grid(True)
    
    # Adding vertical lines for ECG wave components
    plot.axvline(x=0.1, color=royal_blue , linestyle='--', label='P wave')
    plot.axvline(x=0.25, color=crimson , linestyle='--', label='Q wave')
    plot.axvline(x=0.3, color=emerald_green , linestyle='--', label='R wave')
    plot.axvline(x=0.35, color=green , linestyle='--', label='S wave')
    plot.axvline(x=0.6, color=light_blue, linestyle='--', label='T wave')
    
    plot.legend(fontsize=10)
    
    # Create a canvas to display the figure
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
    
def plot_LBBB_ecg():
    # Simulated time axis (500 points)
    time = np.linspace(0, 1, 500)
    
    # Generate synthetic LBBB ECG signal components
    lbbb_signal = (0.1 * np.sin(2 * np.pi * 5 * time) +    # P wave
                   -0.3 * np.exp(-((time - 0.25) ** 2) / 0.002) +  # Q wave
                   1.0 * np.exp(-((time - 0.35) ** 2) / 0.001) +    # R wave (delayed)
                   -0.5 * np.exp(-((time - 0.4) ** 2) / 0.002) +  # S wave
                   0.3 * np.exp(-((time - 0.7) ** 2) / 0.01))      # T wave

    # Create a frame for the plot
    plot_frame = Frame(right_frame, bg="#000")
    plot_frame.pack(side=TOP, fill=BOTH, expand=True)

    # Create a figure for the plot
    fig = Figure(figsize=(5, 2.5), dpi=100)  # Adjust height to 2.5 (50% of 5)
    plot = fig.add_subplot(111)
    plot.plot(time, lbbb_signal, label='LBBB ECG', color=crimson)
    plot.set_title('LBBB ECG Signal', fontsize=12)
    plot.set_xlabel('Time (s)', fontsize=10, color=purpel)
    plot.set_ylabel('Amplitude (mV)', fontsize=10, color=purpel)

    # Change the color of the X-axis and Y-axis lines (spines)
    plot.spines['bottom'].set_color(purpel)  # purpel color for X-axis
    plot.spines['left'].set_color(purpel)    # purpel color for Y-axis
    plot.spines['top'].set_color('none')      # Hide the top spine
    plot.spines['right'].set_color('none') 

    plot.tick_params(axis='x', colors=purpel)  # purpel color for X-axis tick labels
    plot.tick_params(axis='y', colors=purpel)  # purpel color for Y-axis tick labels

    plot.grid(color=purpel)

    # Adding vertical lines for ECG wave components
    plot.axvline(x=0.1, color=royal_blue , linestyle='--', label='P wave')
    plot.axvline(x=0.25, color=crimson , linestyle='--', label='Q wave')
    plot.axvline(x=0.3, color=emerald_green , linestyle='--', label='R wave')
    plot.axvline(x=0.35, color=green , linestyle='--', label='S wave')
    plot.axvline(x=0.6, color=light_blue, linestyle='--', label='T wave')

    plot.legend(fontsize=10)

    # Create a canvas to display the figure
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

root = Tk()
root.title("ECG Signal")
root.geometry("720x480")
root.configure(bg="#2f2f2f")


user_file = None
result = None

def read_file():
    global user_file  # Declare user_file as global
    user_file = filedialog.askopenfilename(filetypes=[("text", "*.txt")])


def Detect_function():
    global result
    result = predict_ecg(user_file)
    

    for widget in right_frame.winfo_children():
        widget.destroy() 
    
    
    plot_Normal_ecg()
    
    if result == "Normal":
        plot_Normal_ecg()
        messagebox.showinfo("Result", "Normal. No disease detected. 🥳انت بتستهبل و صحتك زي الفل🥳")
        
    else:
        plot_LBBB_ecg()
        messagebox.showwarning("Result", "LBBB. 💀اكتب وصيتك💀")
        
        
    

# Frame for the left part
left_frame = Frame(root, bg=left_frame_color, width=400, height=600)
left_frame.pack(side=LEFT, fill=BOTH, expand=True)

# Frame for the right part
right_frame = Frame(root, bg=right_frame_color, width=400, height=600)
right_frame.pack(side=RIGHT, fill=BOTH, expand=True)


read_file_button = Button(left_frame, text="Read file" , command=read_file, bg=button_color, fg="black", font=("Arial", 16))
read_file_button.pack(pady=80)

# Creating the "Detect the state [using KNN]" button
detect_state_button = Button(left_frame, text="Detect the state [using KNN]", command=Detect_function ,  bg=button_color, fg="black", font=("Arial", 16))
detect_state_button.pack(pady=100)

percentage_label = Label(left_frame, text=f"Accuracy: {accuracy_score(y_val, y_val_pred) * 100:.0f}%", bg=left_frame_color, fg=button_color, font=("Arial", 24))
percentage_label.pack(pady=20)




root.mainloop()


