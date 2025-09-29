import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import base64
import io
import joblib
import zipfile
import os


import model_bytes  
import scaler_bytes 

model_zip_data = base64.b64decode(model_bytes.model_b64)
with open("temp_model.zip", "wb") as f:
    f.write(model_zip_data)

with zipfile.ZipFile("temp_model.zip", 'r') as zip_ref:
    zip_ref.extractall("temp_model")

model = load_model("temp_model.keras")

scaler_data = base64.b64decode(scaler_bytes.scaler_b64)
scaler_buf = io.BytesIO(scaler_data)
scaler = joblib.load(scaler_buf)


FEATURES = ['Speed over Ground [knots]', 'Heading [degrees]','Shaft RPM PS  [rpm]', 'Shaft RPM SB [rpm]', 'Shaft Power PS [kW]','Shaft Power SB  [kW]', 'Shaft Torque PS  [kNm]','Shaft Torque SB [kNm]', 'Wind Speed [m/s]']


TIME_STEPS = 3 

model = load_model(r"C:/Users/BrunoNad/Documents/Project_consumption/nn7.keras")


def calculate():
    input_data = []
    for feature in FEATURES:
        feature_vals = []
        for t in range(TIME_STEPS):
            val = entries[(feature, t)].get()
            if val.strip() == "":
                result_label.config(text="Error, all fields must be filled", fg="red")
                return
            try:
                feature_vals.append(float(val))
            except ValueError:
                result_label.config(text=f"Error, invalid input for {feature} at time step {t}", fg="red")
                return
        input_data.append(feature_vals)

    
    data_array = np.array(input_data).T  
    
    
    scaled_array = scaler.transform(data_array)

    
    input_array = scaled_array.reshape((1, TIME_STEPS, len(FEATURES)))

    prediction = model.predict(input_array)
    pred_value = prediction[0][0]

    result_label.config(text=f"Predicted fuel consumption: {pred_value:.4f}", fg="blue")

root = tk.Tk()
root.title("Ship Fuel Consumption Prediction")

entries = {}


tk.Label(root, text="Feature").grid(row=0, column=0, padx=5)
for t in range(TIME_STEPS):
    tk.Label(root, text=f"t-{TIME_STEPS - t}").grid(row=0, column=t + 1, padx=5)


for i, feature in enumerate(FEATURES):
    tk.Label(root, text=feature).grid(row=i + 1, column=0, sticky=tk.W, padx=5, pady=2)
    for t in range(TIME_STEPS):
        entry = tk.Entry(root, width=10)
        entry.grid(row=i + 1, column=t + 1, padx=3, pady=2)
        entries[(feature, t)] = entry

result_label = tk.Label(root, text="", fg="blue", font=("Arial", 12))
result_label.grid(row=len(FEATURES) + 2, column=0, columnspan=TIME_STEPS + 1, pady=10)

calc_button = tk.Button(root, text="Calculate consumption", command=calculate)
calc_button.grid(row=len(FEATURES) + 3, column=0, columnspan=TIME_STEPS + 1, pady=5)

root.mainloop()