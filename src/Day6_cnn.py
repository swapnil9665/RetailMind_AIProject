
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models


# 1. Paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fashion-product-images-small"
IMAGE_DIR = DATA / "images"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# 2. Load dataset
df = pd.read_csv(DATA / "styles.csv", on_bad_lines="skip")

df = df[["id", "masterCategory"]].dropna()
df["id"] = pd.to_numeric(df["id"], errors="coerce")
df = df.dropna(subset=["id"])
df["id"] = df["id"].astype(int)

df["image_path"] = df["id"].apply(
    lambda x: str(IMAGE_DIR / f"{x}.jpg")
)

df = df[df["image_path"].apply(lambda x: Path(x).exists())]

# Keep categories having at least 2 images
counts = df["masterCategory"].value_counts()
df = df[df["masterCategory"].isin(counts[counts >= 2].index)]

categories = sorted(df["masterCategory"].unique())
category_map = {name: i for i, name in enumerate(categories)}

df["label"] = df["masterCategory"].map(category_map)

print("Total images:", len(df))
print("Categories:", categories)


# 3. Split dataset
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=42,
    stratify=train_df["label"]
)


# 4. Image loading function
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label


def create_dataset(data, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (data["image_path"].values, data["label"].values)
    )

    if training:
        ds = ds.shuffle(1000)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = create_dataset(train_df, True)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)


# 5. Image augmentation
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# 6. Build CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    augmentation,

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(categories), activation="softmax")
])


# 7. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# 8. Train model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop]
)


# 9. Evaluate model
loss, accuracy = model.evaluate(test_ds)

print("\nTest Accuracy:", round(accuracy, 4))
print("Test Loss:", round(loss, 4))


# 10. Predictions
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)


# 11. Classification report
print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(categories))),
    target_names=categories,
    zero_division=0
))


# 12. Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(categories)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=categories
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Fashion Product CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(MODEL_DIR / "day6_confusion_matrix.png")
plt.show()


# 13. Save model
model.save(MODEL_DIR / "fashion_cnn.keras")

print("\nCNN model saved successfully!")
