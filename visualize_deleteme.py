#%%
## Visualize the test_predictions.py output

import numpy as np
pred = np.load("my_artefacts/valid_pred.npy")
true = np.load("my_artefacts/valid.npy")

# pred = np.load("my_artefacts/test_pred.npy")
# true = np.load("my_artefacts/test.npy")

print("Shapes:")
print(f"pred: {pred.shape}")
print(f"true: {true.shape}")

## For the first time series, plot the entire sequence, one dimension at a time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
## For the white background interface
## Set the style to whitegrid
sns.set_style("whitegrid")
sns.set_context("talk")
mpl.style.use("bmh")

# nb_features = pred.shape[2]
nb_features = 2

fig, ax = plt.subplots(nb_features, 1, figsize=(10, 5*nb_features), sharex=True)

colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
for i in range(nb_features):
    ax[i].plot(true[0, :, i], label="true", color=colors[i], lw=1)
    ax[i].plot(pred[0, :, i], label="pred", color=colors[i], linestyle="--", lw=3)
    ax[i].set_title(f"Feature {i}")
    ax[i].legend()
    ax[i].grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()


## Calculate the RMSE between the predictions and the true values
pred = pred[:, 96:, :]
true = true[:, 96:, :]

def metrics(pred, true):
    """
    Calculate the MSE, RMSE, MAE between the predictions and the true values
    """
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - true))
    return mse, rmse, mae
mse, rmse, mae = metrics(pred, true)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
