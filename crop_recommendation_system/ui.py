import tkinter as tk
import pandas as pd

from tkinter import messagebox
import joblib

# Load the trained model
model = joblib.load('model.pkl')

def predict_crop():
    try:
        data = [float(entry.get()) for entry in entries]
        columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        input_df = pd.DataFrame([data], columns=columns)
        prediction = model.predict(input_df)[0]
        messagebox.showinfo("Prediction", f"🌾 Recommended Crop: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

window = tk.Tk()
window.title("Crop Recommendation System")

labels = ["Nitrogen (N)", "Phosphorous (P)", "Potassium (K)", 
          "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (mm)"]

entries = []

for i, text in enumerate(labels):
    label = tk.Label(window, text=text)
    label.grid(row=i, column=0, padx=10, pady=5)

    entry = tk.Entry(window)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

button = tk.Button(window, text="Predict Crop", command=predict_crop)
button.grid(row=len(labels), column=0, columnspan=2, pady=10)

window.mainloop()
