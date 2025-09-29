from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# Data Augmentation ve Normalizasyon
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    brightness_range=[0.8, 1.2],  # Düzeltildi
    validation_split=0.1
)

image_datagen = ImageDataGenerator(
    rescale=1./255
)

# Dataset parametreleri
DATA_DIR = "chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
CLASS_MODE = "binary"

# Training generator
train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="training",
    shuffle=True  # Düzeltildi
)

# Validation generator
val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="validation",
    shuffle=False
)

# Test generator
test_gen = image_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False
)

# Sınıf isimleri
class_names = list(train_gen.class_indices.keys())

# Örnek görüntüleri gösterme
images, labels = next(train_gen)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.title(class_names[int(labels[i])])
    plt.axis("off")
plt.show()

base_model = DenseNet121(weights='imagenet',
                        include_top=False, 
                        input_shape=((*IMG_SIZE, 3))
                )
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(optimizer=Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy']    
)

callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='max'),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

print(model.summary())

history = model.fit(
    train_gen,
    epochs=2,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

pred_probs = model.predict(test_gen, verbose=1)
preds = (pred_probs > 0.5).astype(int).ravel()
true_labels = test_gen.classes

cm = confusion_matrix(true_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()