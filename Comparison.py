import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === File paths ===
pred_path = "D:/ISRO Forest Fire/GEE_Train/fire_risk_2022_rf_bin.tif"
truth_path = "D:/ISRO Forest Fire/GEE_Train/FireMask_2022.tif"

# === Load rasters ===
with rasterio.open(pred_path) as src_pred, rasterio.open(truth_path) as src_truth:
    pred = src_pred.read(1)
    truth = src_truth.read(1)
    nodata_pred = src_pred.nodata if src_pred.nodata is not None else -9999
    nodata_truth = src_truth.nodata if src_truth.nodata is not None else -9999

# === Create valid mask ===
valid_mask = (pred != nodata_pred) & (truth != nodata_truth)

# === Flatten filtered arrays ===
y_pred = pred[valid_mask].astype(int)
y_true = truth[valid_mask].astype(int)

# === Optional: Map FireMask if fire=2 instead of 1 ===
# y_true = np.where(y_true == 2, 1, 0)

# === Evaluation Metrics ===
print("✅ Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# === Visualization ===
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# Ground truth
ax[0].imshow(truth, cmap='Reds')
ax[0].set_title("🔥 Ground Truth (FireMask_2022)")
ax[0].axis("off")

# Prediction
ax[1].imshow(pred, cmap='Reds')
ax[1].set_title("🧠 Predicted Fire Mask (Binary)")
ax[1].axis("off")

# Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(ax=ax[2], cmap='Blues', colorbar=False)
ax[2].set_title("📊 Confusion Matrix")

plt.tight_layout()
plt.show()

with rasterio.open("D:/ISRO Forest Fire/GEE_Train/fire_risk_2022_rf_prob.tif") as src:
    prob = src.read(1)

plt.imshow(prob, cmap='hot', vmin=0, vmax=1)
plt.title("🔥 Fire Risk Probability (2022)")
plt.colorbar(label="Fire Probability")
plt.axis("off")
plt.show()
