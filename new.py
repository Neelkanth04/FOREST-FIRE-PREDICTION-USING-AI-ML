import rasterio
import matplotlib.pyplot as plt
import numpy as np

# === Paths ===
base = "D:/ISRO Forest Fire/GEE_Train/"
firemask_path = base + "FireMask_2022.tif"
prediction_path = base + "fire_risk_2022_rf_bin.tif"  # or fire_risk_2022_bin.tif

# === Load data ===
with rasterio.open(firemask_path) as src:
    firemask = src.read(1)
with rasterio.open(prediction_path) as src:
    prediction = src.read(1)

# === Clean invalid pixels (optional) ===
firemask = np.where((firemask == 0) | (firemask == 1), firemask, np.nan)
prediction = np.where((prediction == 0) | (prediction == 1), prediction, np.nan)

# === Plot ===
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("🔥 Ground Truth (FireMask 2022)")
plt.imshow(firemask, cmap="Reds")
plt.colorbar(label="Fire (1) / No Fire (0)")

plt.subplot(1, 3, 2)
plt.title("🧠 Predicted Fire (Threshold 0.05)")
plt.imshow(prediction, cmap="YlOrRd")
plt.colorbar(label="Predicted Fire (1)")

plt.subplot(1, 3, 3)
plt.title("🔍 Overlay")
overlay = np.zeros((*prediction.shape, 3), dtype=np.uint8)
overlay[:, :, 0] = (firemask == 1) * 255       # Red = Fire in ground truth
overlay[:, :, 1] = (prediction == 1) * 255     # Green = Fire in prediction
plt.imshow(overlay)
plt.axis('off')
plt.title("Red = GT Fire, Green = Predicted")

plt.tight_layout()
plt.show()
