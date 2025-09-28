"""
    Flowers Datasets:
        rgb 224x224
    Cnn ile sınıflandırma modeli oluşturup problemi çözme
"""

from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt


(ds_train,ds_val),ds_info = load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
)

print(ds_info.features)
print(ds_info.features["label"].num_classes)

fig = plt.figure(figsize=(10,5))
for i, (image, label) in enumerate(ds_train.take(6)):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(image)
    ax.set_title(ds_info.features["label"].int2str(label))
    ax.axis("off")
plt.show()

IMG_SIZE = (180, 180)

def preprocess_train(image,label):
    image = tf.image.resize(image,IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image,max_delta=0.1)
    image = tf.image.random_contrast(image,lower=0.9,upper=1.2)
    image = tf.image.random_crop(image,size = (160,160,3))
    image = tf.image.resize(image,IMG_SIZE)
    image = tf.cast(image,tf.float32) / 255.0
    return image,label

def preprocess_val(image,label):
    image = tf.image.resize(image,IMG_SIZE)
    image = tf.cast(image,tf.float32) / 255.0
    return image,label
    
ds_train = (
    ds_train.map(preprocess_train,
                num_parallel_calls=AUTOTUNE).shuffle(1000).batch(32).prefetch(AUTOTUNE
    
))

ds_val = (
    ds_val.map(preprocess_val,
                num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
)

model = Sequential([
    
    # Feature Extraction
    Conv2D(32,(3,3),activation="relu",input_shape=(*IMG_SIZE,3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D((2,2)),
                             
    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D((2,2)),
                           
    # Classification
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.5),
    Dense(ds_info.features["label"].num_classes,activation="softmax"),
])


callbacs = [
    EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=2,min_lr=1e-6,verbose=1),
    ModelCheckpoint("best_model.h5",save_best_only=True),
]

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=20,
    callbacks=callbacs,
)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Train Accuracy")
plt.plot(history.history["val_accuracy"],label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"],label="Train Loss")
plt.plot(history.history["val_loss"],label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


