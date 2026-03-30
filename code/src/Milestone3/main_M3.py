import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from google.colab import files

# ----------------------------
# Load dataset
# ----------------------------

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
# Step 6: Final VRP
# ----------------------------

data["Final_VRP"] = data["Final_IV"]**2 - data["Final_RV"]**2

# ----------------------------
# Visualization
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(data["Final_VRP"])
plt.title("Variance Risk Premium with Measurement Error")
plt.xlabel("Time")
plt.ylabel("VRP")
plt.grid(True)
plt.show()

# ----------------------------
# Plot 2: Empirical PDF (M3)
# ----------------------------

plt.figure()
plt.hist(data["Final_VRP"], bins=30, density=True)
plt.title("Empirical PDF of VRP (M3)")
plt.xlabel("VRP")
plt.ylabel("Density")
plt.show()

# ----------------------------
# Plot 3: Empirical CDF (M3)
# ----------------------------

sorted_vrp = np.sort(data["Final_VRP"])
cdf = np.arange(1, len(sorted_vrp) + 1) / len(sorted_vrp)

plt.figure()
plt.plot(sorted_vrp, cdf)
plt.title("Empirical CDF of VRP (M3)")
plt.xlabel("VRP")
plt.ylabel("Cumulative Probability")
plt.show()
