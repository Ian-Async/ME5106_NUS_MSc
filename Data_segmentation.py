import scipy.io as sio
import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))
mat_path = os.path.join(base, "ExperimentData", "SingleTrack_Division.mat")
output_folder = os.path.join(base, "Division")

os.makedirs(output_folder, exist_ok=True)

mat_data = sio.loadmat(mat_path)

prefix_list = ["A", "B", "C"]

for prefix in prefix_list:
    for i in range(1, 15):
        var_name = f"{prefix}{i}"
        if var_name in mat_data:
            df = pd.DataFrame(mat_data[var_name])
            df.to_csv(os.path.join(output_folder, f"SingleTrack_Division_{var_name}.csv"), index=False)