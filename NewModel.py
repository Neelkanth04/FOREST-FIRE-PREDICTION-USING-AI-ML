import os
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

# Paths
ROOT = r"D:/ISRO Forest Fire/GEE_Train"
X_PATH = os.path.join(ROOT, "X_train.npy")
Y_PATH = os.path.join(ROOT, "y_train.npy")

# Load data as memory‐mapped arrays
X = np.load(X_PATH, mmap_mode='r')
y = np.load(Y_PATH, mmap_mode='r')
n_samples, n_features = X.shape
print(f"✅ Loaded dataset: {n_samples:,} samples, {n_features} features")

# Chunk settings
CHUNK_SIZE = 10_000_000
n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"🔁 Will train on {n_chunks} chunks of up to {CHUNK_SIZE:,} rows each")

for chunk_idx in range(n_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_samples)
    X_chunk = X[start:end]
    y_chunk = y[start:end]
    unique_classes = np.unique(y_chunk)

    # Skip if only one class present
    if unique_classes.size < 2:
        print(f"⚠️ Chunk {chunk_idx+1}/{n_chunks} skipped (only class {unique_classes[0]})")
        continue

    # Compute class weights for this chunk
    counts = np.bincount(y_chunk)
    total = counts.sum()
    class_weights = {
        0: total / (2 * counts[0]),
        1: total / (2 * counts[1])
    }
    print(f"📦 Chunk {chunk_idx+1}/{n_chunks} → samples={end-start:,}, class_weights={class_weights}")

    # Initialize and train model on this chunk
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=10,
        early_stopping=True,
        class_weight=class_weights,
        random_state=42 + chunk_idx
    )
    model.fit(X_chunk, y_chunk)

    # Save the chunk model
    model_filename = f"hgb_chunk_{chunk_idx+1:02d}.pkl"
    model_path = os.path.join(ROOT, model_filename)
    joblib.dump(model, model_path)
    print(f"✅ Saved model for chunk {chunk_idx+1} → {model_filename}")

print("🎉 All chunk models trained and saved.")
