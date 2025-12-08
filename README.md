# Facial Emotion Classification with CNN

Convolutional neural network for **7-class facial emotion recognition** on **48×48 grayscale images** using the FER Kaggle dataset (`ananthu017/emotion-detection-fer`).  
The project includes **data loading from KaggleHub, data augmentation, class imbalance handling with class weights, training, and evaluation**.

---

## 1. Problem Overview

The goal of this project is to classify facial expressions into the following **7 emotion classes**:

- `angry`
- `disgusted`
- `fearful`
- `happy`
- `neutral`
- `sad`
- `surprised`

Input images are **48×48, grayscale**, and the model is trained using a **Convolutional Neural Network (CNN)** implemented with **TensorFlow/Keras**.

---

## 2. Dataset

This project uses the Kaggle dataset:

> `ananthu017/emotion-detection-fer`

It is automatically downloaded using **KaggleHub**:

- Train directory: `.../emotion-detection-fer/train`
- Test directory: `.../emotion-detection-fer/test`

The training directory is further split into:

- **Train**: 80% of `train/`
- **Validation**: 20% of `train/`
- **Test**: 100% of `test/`

Images are loaded using:

```python
tf.keras.utils.image_dataset_from_directory(
    ...,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
)
```

---

## 3. Main Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- KaggleHub

---

## 4. Project Structure (suggested)

```text
facial-emotion-classification-cnn/
├── notebook.ipynb            # Main training & experimentation notebook
├── README.md
├── requirements.txt          # (Optional) Python dependencies
└── models/                   # (Optional) Saved models
    └── best_model.h5
```

You can keep everything inside a Jupyter/Colab notebook or later refactor into Python scripts.

---

## 5. How to Run

### 5.1. Requirements

Install the dependencies (example):

```bash
pip install tensorflow kagglehub numpy matplotlib scikit-learn
```

On Google Colab, most of these are pre-installed. Only `kagglehub` may require installation:

```python
!pip install -q kagglehub
```

### 5.2. Download the Dataset

The dataset is downloaded via KaggleHub:

```python
import kagglehub
import os

path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")

train_dir = os.path.join(path, "train")
test_dir  = os.path.join(path, "test")
```

### 5.3. Load Datasets

```python
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    shuffle=False,
    batch_size=BATCH_SIZE,
)
```

---

## 6. Data Pipeline & Class Weights

Datasets are prepared with shuffling, caching and prefetching:

```python
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False, cache=False):
    if shuffle:
        ds = ds.shuffle(5000)
    if cache:
        ds = ds.cache()
    return ds.prefetch(AUTOTUNE)

train_ds_p = prepare(train_ds, shuffle=True,  cache=False)
val_ds_p   = prepare(val_ds,   shuffle=False, cache=True)
test_ds_p  = prepare(test_ds,  shuffle=False, cache=True)
```

To deal with **class imbalance**, class weights are computed from the label distribution in the training set:

```python
from collections import Counter

all_labels = []
for _, labels in train_ds:
    all_labels.extend(labels.numpy())

counter = Counter(all_labels)
total = sum(counter.values())

class_weight = {
    int(cls_id): float(total / (len(counter) * count))
    for cls_id, count in counter.items()
}
```

These `class_weight`s are then passed to `model.fit(...)`.

---

## 7. Model Architecture

The model is a **CNN** with three convolutional blocks and global average pooling:

- Input: `(48, 48, 1)` grayscale
- Data augmentation:
  - Random horizontal flip
  - Small rotations
  - Random zoom
  - Small translations
- Conv blocks:
  - 2×Conv2D + BatchNorm + ReLU + MaxPooling2D (filters: 32 → 64 → 128)
- GlobalAveragePooling2D
- Dense(128, ReLU) + Dropout
- Output: Dense(7, softmax)

Simplified code:

```python
def build_model(img_size, num_classes, learning_rate=1e-3):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ])

    inputs = keras.Input(shape=img_size + (1,))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    # Conv Block 1
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Conv Block 2
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    # Conv Block 3
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

---

## 8. Training

Training uses:

- `EarlyStopping` (monitoring `val_loss`, patience 10)
- `ReduceLROnPlateau` (monitoring `val_loss`, factor 0.5, patience 5)
- `class_weight` to compensate class imbalance

Example:

```python
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1,
)

model = build_model(IMG_SIZE, num_classes, LEARNING_RATE)

history = model.fit(
    train_ds_p,
    validation_data=val_ds_p,
    epochs=50,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight,
)
```

---

## 9. Results

On the held-out **test set**, the model achieves approximately:

- **Test accuracy** ≈ **0.58**
- **Macro F1-score** ≈ **0.54**

Per-class performance (example run):

- `happy` and `surprised`: F1 ≈ **0.82** and **0.72**
- `neutral` and `angry`: F1 around **0.50–0.55**
- `disgusted` (very imbalanced in the dataset): F1 ≈ **0.45** thanks to class weighting
- `fearful` and `sad`: lower F1 (≈0.35–0.41), often confused with other negative emotions

A confusion matrix is also plotted to visualize misclassifications between emotions.

---

## 10. Future Work

Possible improvements:

- Try label smoothing (with one-hot labels and `CategoricalCrossentropy`)
- Oversampling or targeted augmentation for difficult classes (e.g., `fearful`, `sad`)
- Experiment with deeper CNNs or pretrained backbones
- Add experiment tracking and multiple runs comparison

---

