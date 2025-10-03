import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

stupci = ['Speed over Ground [knots]', 'Heading [degrees]', 'Shaft RPM PS [rpm]','Shaft RPM SB [rpm]', 'Shaft Power PS [kW]', 'Shaft Power SB [kW]','Shaft Torque PS [kNm]', 'Shaft Torque SB [kNm]', 'Wind Speed [m/s]',
          'Consumption 10 minutes ago', 'Consumption 20 minutes ago','Consumption 30 minutes ago', 'Consumption 40 minutes ago','Consumption 50 minutes ago']
          
koeficijenti = [291.48583616784373, -3.211413, 0.014851, -0.927242, 1.153701, 0.437464,0.142714, 0.298328, -0.752784, -2.352980, 0.002621, -0.013778, 0.004272,-0.009622, -0.010273]
                
data = None

def load_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        try:
            df = pd.read_excel(file_path)
            missing_cols = [col for col in stupci if col not in df.columns]
            if missing_cols:
                messagebox.showerror("Error", f"Missing columns in Excel file: {missing_cols}")
                return
            data = df[stupci]
            calc_button.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Excel file:\n{e}")

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

load_button = tk.Button(root, text="Load Excel File", command=load_file)
load_button.pack(pady=10)

calc_button = tk.Button(root, text="Calculate", command=calculate, state=tk.DISABLED)
calc_button.pack(pady=10)

result_text = tk.Text(root, height=15, width=50)
result_text.pack(pady=10)

root.mainloop()