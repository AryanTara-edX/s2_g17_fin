import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from google.colab import files

try:
    data = pd.read_csv("vrp_data_final.csv")
except FileNotFoundError:
    print("File 'vrp_data_final.csv' not found. Please upload it below:")
    uploaded = files.upload()
    if 'vrp_data_final.csv' in uploaded:
        data = pd.read_csv(io.BytesIO(uploaded['vrp_data_final.csv']))
    else:
        raise FileNotFoundError("The required file 'vrp_data_final.csv' was not uploaded.")

# ----------------------------
# Step 1: Log returns
# ----------------------------

data["returns"] = np.log(data["Closed for RV(Pt)"] / data["P(t-1)"])

# ----------------------------
# Step 2: Realized volatility
# ----------------------------

data["RV"] = data["Closed for RV(Pt)"] / 100

# ----------------------------
# Step 3: Implied volatility
# ----------------------------

data["IV"] = data["Close for VIX"] / 100

# ----------------------------
# Step 4: Variance Risk Premium
# ----------------------------

data["VRP"] = data["IV"]**2 - data["RV"]**2

# ----------------------------
# Step 5: Measurement error model
# ----------------------------

data["Final_IV"] = data["true volatality"] + data["IV_error"]
data["Final_RV"] = data["true volatality"] + data["RV_error"]

# ----------------------------
# Step 6: Final VRP (M3 output)
# ----------------------------

data["Final_VRP"] = data["Final_IV"]**2 - data["Final_RV"]**2

# ----------------------------
# Step 7: Monte Carlo sim of noise (M4)
# ----------------------------

num_sim = 10000
simulated_vrp = []

true_vol = data["true volatality"].values
n = len(true_vol)

for _ in range(num_sim):

    iv_error = np.random.normal(0, np.std(data["IV_error"]), n)
    rv_error = np.random.normal(0, np.std(data["RV_error"]), n)

    IV_sim = true_vol + iv_error
    RV_sim = true_vol + rv_error

    VRP_sim = IV_sim**2 - RV_sim**2

    simulated_vrp.extend(VRP_sim)

# ----------------------------
# Plot 1: M4 PDF
# ----------------------------

plt.figure()
plt.hist(simulated_vrp, bins=30, density=True)
plt.title("PDF of VRP (Monte Carlo M4)")
plt.xlabel("VRP")
plt.ylabel("Density")
plt.show()

# ----------------------------
# Plot 2: M4 CDF
# ----------------------------

sorted_data = np.sort(simulated_vrp)
cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)

plt.figure()
plt.plot(sorted_data, cdf)
plt.title("CDF of VRP (Monte Carlo M4)")
plt.xlabel("VRP")
plt.ylabel("Cumulative Probability")
plt.show()
