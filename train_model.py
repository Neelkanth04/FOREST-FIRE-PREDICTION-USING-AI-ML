import os
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import psutil
from tqdm import tqdm
from prepare_yeardata import prepare_year_stack, load_label, slope  # Your custom module

# ---- Configuration ----
root = r'D:\ISRO Forest Fire\GEE_Train'
years = [2017, 2018, 2019, 2020, 2021]
patch_size = 256
max_patches_per_year = 100  # adjust as needed for RAM
print_every_n = 10  # show memory status every N patches

# ---- Progress-safe containers ----
X_all, y_all = [], []

def print_memory():
    mem = psutil.virtual_memory()
    print(f"🧠 RAM used: {mem.used // (1024**2)} MB / {mem.total // (1024**2)} MB ({mem.percent}%)")

# ---- Training Loop ----
for year in years:
    print(f"\n📅 Sampling patches for year {year}...")
    features = prepare_year_stack(year, slope)
    labels = load_label(year, slope)

    if features is None or labels is None:
        print(f"⚠️ Skipping year {year} due to missing data.")
        continue

    H, W, C = features.shape
    patch_count = 0

    # Shuffle starting points
    ys = np.arange(0, H, patch_size)
    xs = np.arange(0, W, patch_size)
    np.random.shuffle(ys)
    np.random.shuffle(xs)

    with tqdm(total=max_patches_per_year, desc=f"Year {year}", unit="patch") as pbar:
        for y_off in ys:
            for x_off in xs:
                if patch_count >= max_patches_per_year:
                    break

                height = min(patch_size, H - y_off)
                width = min(patch_size, W - x_off)

                if height <= 0 or width <= 0:
                    continue

                patch_feat = features[y_off:y_off + height, x_off:x_off + width, :]
                patch_label = labels[y_off:y_off + height, x_off:x_off + width]

                flat_feat = patch_feat.reshape(-1, C)
                flat_label = patch_label.flatten()

                valid = ~np.any(np.isnan(flat_feat), axis=1) & (flat_label >= 0)
                if np.sum(valid) == 0:
                    continue

                X_all.append(flat_feat[valid])
                y_all.append(flat_label[valid])

                patch_count += 1
                pbar.update(1)

                if patch_count % print_every_n == 0:
                    print_memory()

# ---- Final Training Data ----
print("\n📦 Stacking all patch data...")
X_train = np.vstack(X_all)
y_train = np.hstack(y_all)
print(f"✅ Training dataset: {X_train.shape}, Labels: {y_train.shape}")
print_memory()

# ---- Model Training ----
print("\n🧠 Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("✅ Model training complete.")

# ---- Save Model ----
model_path = os.path.join(root, "fire_risk_model.pkl")
joblib.dump(clf, model_path)
print(f"💾 Model saved to: {model_path}")

# ---- Accuracy Report ----
y_pred = clf.predict(X_train)
print("\n📊 Training classification report:")
print(classification_report(y_train, y_pred))
