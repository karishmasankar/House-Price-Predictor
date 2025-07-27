import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load the model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create window
window = tk.Tk()
window.title("🏠 House Price Predictor")
window.geometry("500x400")
window.configure(bg="#f0f8ff")

title = tk.Label(window, text="House Price Predictor", font=("Arial", 20, "bold"), bg="#f0f8ff", fg="#333")
title.pack(pady=10)

features = ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms"]
entries = []

for feature in features:
    frame = tk.Frame(window, bg="#f0f8ff")
    frame.pack(pady=5)
    
    label = tk.Label(frame, text=feature + ":", font=("Arial", 12), bg="#f0f8ff")
    label.pack(side=tk.LEFT)
    
    entry = tk.Entry(frame, font=("Arial", 12))
    entry.pack(side=tk.LEFT, padx=10)
    
    entries.append(entry)

def predict():
    try:
        input_data = [float(e.get()) for e in entries]
        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]
        messagebox.showinfo("Prediction", f"Estimated House Price: ${prediction:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", "Please enter valid numeric inputs.")

predict_btn = tk.Button(window, text="Predict", command=predict, font=("Arial", 14), bg="#007acc", fg="white", padx=10, pady=5)
predict_btn.pack(pady=20)

window.mainloop()
