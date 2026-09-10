from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = Path(
    "data/xray_dataset/Digital Knee X-ray Images/Digital Knee X-ray Images/Knee X-ray Images/MedicalExpert-I"
)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xray_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 10

class_names = [
    "0Normal",
    "1Doubtful",
    "2Mild",
    "3Moderate",
    "4Severe"
]

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR.resolve()}")

for class_name in class_names:
    if not (DATA_DIR / class_name).exists():
        raise FileNotFoundError(
            f"Missing class folder: {(DATA_DIR / class_name).resolve()}"
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

MODEL_DIR.mkdir(exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]

print("\nStarting X-ray model training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_PATH)

loss, accuracy = model.evaluate(val_ds, verbose=0)

print("\nTraining completed.")
print(f"Model saved at: {MODEL_PATH.resolve()}")
print(f"Validation accuracy: {accuracy:.4f}")