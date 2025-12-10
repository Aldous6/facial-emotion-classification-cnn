# Facial Emotion Classification with CNN

Convolutional neural network for **7-class facial emotion recognition** on **48×48 grayscale images** using the FER Kaggle dataset (`ananthu017/emotion-detection-fer`).  
The project includes **data loading from KaggleHub, data augmentation, class imbalance handling with class weights and oversampling, advanced loss functions (Focal Loss), training, and evaluation**.

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

## 6. Data Pipeline, Class Weights & Oversampling

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

### 6.1. Base Class Weights

To deal with **class imbalance**, initial class weights are computed from the label distribution in the training set:

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

These `class_weight`s can be passed to `model.fit(...)`.

### 6.2. Targeted Oversampling for "Disgusted"

During experimentation, we observed that the `disgusted` class represented **< 2%** of the training data and was extremely hard to learn. Class weights alone were not sufficient.

To address this, we implemented **targeted oversampling**:

- A custom augmentation script generates **synthetic images only for the `disgusted` class**.
- The number of physical samples for `disgusted` was increased from **≈436** to **≈8,000**.
- Augmentations are **conservative** (max rotation 15°, zoom 0.15), preserving subtle mouth/nose geometry that distinguishes `disgusted` from `angry` on low-resolution (48×48) faces.

A critical bug was also fixed: an earlier version of the generator risked creating thousands of copies of a **single** image, leading to immediate overfitting. The logic was corrected so that augmentation is evenly distributed across all original `disgusted` images.

### 6.3. Updated Class Weights After Oversampling

After oversampling, **class weights were recomputed** based on the **new** class distribution:

- `disgusted` becomes more frequent and may even receive a weight **< 1.0**.
- However, it is still treated as a **difficult class** via the loss function (see Focal Loss below), which heavily penalizes misclassifications.

This separation of concerns:

- **Data engineering** (oversampling) ⟶ better signal.
- **Loss design** (Focal Loss) ⟶ gradient focuses on hard examples.

---

## 7. Model Architectures

This project includes a **baseline CNN** and an **improved, deeper VGG-style CNN**.

### 7.1. Baseline CNN

The baseline model is a **CNN with three convolutional blocks** and global average pooling:

- Input: `(48, 48, 1)` grayscale
- Data augmentation:
  - Random horizontal flip
  - Small rotations
  - Random zoom
  - Small translations
- Conv blocks:
  - 2×Conv2D + BatchNorm + ReLU + MaxPooling2D  
    (filters: 32 → 64 → 128)
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

### 7.2. Improved VGG-Style CNN with Strong Regularization

To better capture subtle facial patterns and handle noisy labels, we built a **deeper VGG-style architecture** with:

- **4 convolutional blocks**, with increasing filters:  
  `64 → 128 → 256 → 512`
- Each block:
  - 2×Conv2D (with L2 kernel regularization, no bias)
  - BatchNormalization
  - ReLU
  - MaxPooling2D
  - Optional `SpatialDropout2D` for stronger regularization
- Top:
  - GlobalAveragePooling2D
  - Dense(256) + Dropout
  - Dense(128) + Dropout
  - Dense(7, softmax)

Additional details:

- **L2 Kernel Regularization** on all Conv and Dense layers to reduce overfitting on noisy labels.
- **Spatial Dropout** drops entire feature maps instead of individual neurons, forcing the network to rely on redundant facial structures instead of single pixels.
- **Batch Normalization** after each convolution to stabilize training and allow higher learning rates.
- **Correct pre-processing order**: `Rescaling` is applied before any `GaussianNoise`, so the noise is meaningful in normalized pixel space.

---

## 8. Training

### 8.1. Baseline Training

Baseline training uses:

- `EarlyStopping` (monitoring `val_loss`, patience = 10, restore best weights)
- `ReduceLROnPlateau` (monitoring `val_loss`, factor = 0.5, patience = 5, `min_lr=1e-5`)
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

### 8.2. Advanced Training: Focal Loss & Longer Schedules

To better focus on **hard and underrepresented examples**, we adopted **Focal Loss** in the improved setup:

- Loss: Focal Loss with `gamma = 2.0`, `alpha = 0.25`.
- Intuition:
  - Down-weights **easy** samples that the model already classifies correctly.
  - Up-weights **hard** samples such as `disgusted` and `fearful`, where the model tends to struggle.

In addition:

- Class weights are recomputed **after oversampling**.
- `EarlyStopping` and `ReduceLROnPlateau` use **larger patience**, allowing the model to escape local minima and refine with very small learning rates (down to `1e-6`) in late epochs.

---

## 9. Evaluation Protocol & Data Leakage Prevention

During the process, we discovered a **potential data leakage** pattern:

- The validation set could contain images that were heavily augmented versions of training samples, leading to over-optimistic validation metrics.

To obtain **honest metrics**:

- The main performance numbers reported below are computed on the **held-out test set**, which:
  - Is never used during training.
  - Is not seen by the augmentation routines.
  - Represents “pure” unseen data.

A confusion matrix is also plotted to visualize misclassifications and correlations between emotions (e.g., `sad` vs `fearful`, `angry` vs `disgusted`).

---

## 10. Results

### 10.1. Baseline CNN

On the held-out **test set**, the baseline CNN achieves approximately:

- **Test accuracy** ≈ **0.58**
- **Macro F1-score** ≈ **0.54**

Per-class performance (example run):

- `happy` and `surprised`: F1 ≈ **0.82** and **0.72**
- `neutral` and `angry`: F1 around **0.50–0.55**
- `disgusted` (very imbalanced in the dataset): F1 ≈ **0.45** thanks to class weighting
- `fearful` and `sad`: lower F1 (≈0.35–0.41), often confused with other negative emotions

### 10.2. Improved VGG-Style CNN with Oversampling + Focal Loss

After applying all the improvements (oversampling, deeper CNN, strong regularization and Focal Loss), the model evolves from a “guesser” into a more **analytical** classifier:

- **Test accuracy**: **≈ 0.60–0.61** on the pure test set.  
  This is competitive for a model trained **from scratch** on this dataset, without any pretrained backbone.

**The main qualitative improvement is in the `disgusted` class:**

- **Before (baseline)**:
  - Recall for `disgusted` ≈ **0.27**
  - Most `disgusted` faces were misclassified as `angry`.
- **After improvements**:
  - Recall for `disgusted` ≈ **0.55** (≈ 2× improvement).

This indicates that the model has learned to **separate the morphology of disgust (e.g. wrinkled nose)** from **anger (e.g. frowning eyebrows)**, even at 48×48 resolution. In other words, the symmetry in error between these two negative emotions is broken.

**Trade-offs:**

- The recall for `angry` decreases somewhat.
- The model becomes more conservative: when uncertain between `angry` and another class, it often prefers `neutral`.
- This is an expected behavior when correcting a major bias: the system finds a new balance between sensitivity and specificity across negative emotions.

**Technical takeaway:**

> The final system is not just a CNN that “memorizes” the dataset.  
> It explicitly addresses a **physical signal problem** (extreme class imbalance and noisy labels) through:
> - targeted data engineering,
> - advanced loss functions, and
> - strong regularization.  
> The result is a more robust and balanced facial emotion recognizer.

---

## 11. Future Work

Possible next steps:

- Try **label smoothing** (with one-hot labels and `CategoricalCrossentropy`) on top of Focal Loss experiments for comparison.
- Explore **more advanced oversampling strategies** or class-specific augmentations for `fearful` and `sad`.
- Replace the from-scratch CNN with:
  - Pretrained feature extractors (e.g. ResNet, EfficientNet) adapted to 48×48 grayscale, or
  - Vision Transformers for small images.
- Add **experiment tracking** (Weights & Biases, TensorBoard, etc.) to systematically compare multiple runs and hyperparameters.
- Investigate **curriculum learning**: start training on “easy” examples and gradually add harder samples.
