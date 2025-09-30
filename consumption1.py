import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox


def calculate_consumption():
    try:
        values = [float(entries[col].get()) for col in coefficients.keys()]
        consumption = sum(v * coefficients[col] for v, col in zip(values, coefficients.keys())) + intercept
        messagebox.showinfo("Result", f'Calculated fuel consumption (L/hr): {consumption:.2f}')
    except ValueError:
        messagebox.showerror("Input Error", "All fields must be filled.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

stupci=['Speed over Ground [knots]','Heading [degrees]','Shaft RPM PS [rpm]','Shaft RPM SB [rpm]','Shaft Power PS [kW]','Shaft Power SB [kW]','Shaft Torque PS [kNm]',
         'Shaft Torque SB [kNm]','Wind Speed [m/s]','Consumption 10 minutes ago','Consumption 20 minutes ago','Consumption 30 minutes ago','Consumption 40 minutes ago','Consumption 50 minutes ago']
         
koeficijenti=[-3.211413,0.014851,-0.927242,1.153701,0.437464,0.142714,0.298328,-0.752784,-2.352980,0.002621,-0.013778,0.004272,-0.009622,-0.010273]

coefficients=dict(zip(stupci,koeficijenti))

intercept=291.48583616784373

root = tk.Tk()
root.title("Consumption Calculator")

entries = {}

for i, col in enumerate(coefficients.keys()):
    label = tk.Label(root, text=col, fg="blue", anchor="w")
    label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=2)
    entries[col] = entry

def calculate_consumption():
    try:
        values = [float(entries[col].get()) for col in coefficients.keys()]
        consumption = sum(v * coefficients[col] for v, col in zip(values, coefficients.keys())) + intercept
        result_label.config(text=f'Consumption (L/hr): {consumption:.2f}', fg="black")
    except ValueError:
        result_label.config(text="All fields must be filled.", fg="red")
    except Exception as e:
        result_label.config(text=str(e), fg="red")

button = tk.Button(root, text="Calculate consumption", bg="red", fg="white", command=calculate_consumption)
button.grid(row=len(coefficients), column=0, columnspan=2, pady=10, ipadx=10, ipady=5)

result_label = tk.Label(root, text="", fg="green", anchor="w")
result_label.grid(row=len(coefficients)+1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

root.mainloop()
