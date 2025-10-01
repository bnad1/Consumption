import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

stupci = ['Speed over Ground [knots]', 'Heading [degrees]', 'Shaft RPM PS [rpm]',
          'Shaft RPM SB [rpm]', 'Shaft Power PS [kW]', 'Shaft Power SB [kW]', 
          'Shaft Torque PS [kNm]', 'Shaft Torque SB [kNm]', 'Rate of turn','Wind Speed [m/s]',
          'Consumption 10 minutes ago', 'Consumption 20 minutes ago',
          'Consumption 30 minutes ago', 'Consumption 40 minutes ago',
          'Consumption 50 minutes ago']

koeficijenti = [287.2863130821273,-2.795556,0.022874,-0.918029,1.221253,0.444881,0.144284,0.248228,-0.849692,-0.407847,-1.904554,-0.003768,-0.008412,0.003764,-0.021995,-0.002580]

data = None

def load_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            missing_cols = [col for col in stupci if col not in df.columns]
            if missing_cols:
                messagebox.showerror("Error", f"Missing columns in CSV file: {missing_cols}")
                return
            data = df[stupci]
            #file_label.config(text=f"Loaded file: {file_path}")
            calc_button.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")

def calculate():
    if data is None:
        messagebox.showwarning("Warning", "No data loaded")
        return
    intercept = koeficijenti[0]
    coefs = koeficijenti[1:]
    results = []
    for index, row in data.iterrows():
        prediction = intercept + sum(row[col] * coef for col, coef in zip(stupci, coefs))
        results.append((index + 1, prediction))

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Row\tPredicted Fuel Consumption\n")
    result_text.insert(tk.END, "-" * 35 + "\n")
    for idx, val in results:
        result_text.insert(tk.END, f"{idx}\t{val:.4f}\n")

root = tk.Tk()
root.title("Fuel Consumption Calculator")

load_button = tk.Button(root, text="Load CSV File", command=load_file)
load_button.pack(pady=10)

#file_label = tk.Label(root, text="No file loaded")
#file_label.pack(pady=5)

calc_button = tk.Button(root, text="Calculate", command=calculate, state=tk.DISABLED)
calc_button.pack(pady=10)

result_text = tk.Text(root, height=15, width=50)
result_text.pack(pady=10)

root.mainloop()