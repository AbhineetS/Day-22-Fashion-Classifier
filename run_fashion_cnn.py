# run_fashion_cnn.py
"""
Day 22 — Fashion Item Classifier (simple CNN)
Saves model, training history plot, and sample predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Settings
MODEL_OUT = "fashion_cnn.h5"
HISTORY_IMG = "training_history.png"
SAMPLES_IMG = "sample_predictions.png"
EPOCHS = 6
BATCH_SIZE = 128

def load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # normalize & reshape
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test

def build_model(input_shape=(28,28,1), n_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def plot_history(history, fname=HISTORY_IMG):
    plt.figure(figsize=(8,4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_sample_predictions(model, x_test, y_test, fname=SAMPLES_IMG, n=12):
    class_names = [
        "T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"
    ]
    idx = np.random.choice(len(x_test), n, replace=False)
    x = x_test[idx]
    y_true = y_test[idx]
    preds = model.predict(x).argmax(axis=1)

    cols = 6
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*2, rows*2.2))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x[i].squeeze(), cmap="gray")
        plt.title(f"T:{class_names[y_true[i]]}\nP:{class_names[preds[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def main():
    print("📦 Loading and preprocessing data...")
    x_train, y_train, x_test, y_test = load_and_preprocess()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    print("🧠 Building model...")
    model = build_model(input_shape=x_train.shape[1:])
    model.summary()

    print(f"🚀 Training for {EPOCHS} epochs...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.12,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )

    print("📈 Saving training plot...")
    plot_history(history, HISTORY_IMG)

    print(f"💾 Saving model to {MODEL_OUT} ...")
    model.save(MODEL_OUT)

    print("🔍 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

    print("🧾 Classification report")
    y_pred = model.predict(x_test).argmax(axis=1)
    print(classification_report(y_test, y_pred, digits=4))

    # confusion matrix image
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("💾 Saved confusion_matrix.png")

    print("🖼️ Saving sample predictions...")
    save_sample_predictions(model, x_test, y_test, SAMPLES_IMG)

    print("✅ Done.")

if __name__ == "__main__":
    main()