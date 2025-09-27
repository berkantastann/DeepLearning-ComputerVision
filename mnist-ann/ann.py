"""
    Mnist veri seti:
    0-9 arası el yazısı rakamların bulunduğu bir veri setidir.
    60.000 eğitim ve 10.000 test örneği içerir.
    Her örnek 28x28 piksel boyutunda gri tonlamalı bir görüntüdür.
    Her görüntü, 0-9 arasındaki bir rakamı temsil eder.
    Veri seti, makine öğrenimi ve bilgisayarla görme alanlarında yaygın olarak kullanılır.
    
    Image processing:
    histogram eşitleme: kontrast arttırma
    gürültü giderme: gaussian blur
    cany edge edge detection: kenar tespiti    
    
    ANN ile mnist veri seti sınıflandırma:
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print("x_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)

 
# Görüntü işleme

img = x_train[4]

stages = {"original": img}

eq = cv.equalizeHist(img)
stages["histogram equalization"] = eq

blur = cv.GaussianBlur(eq,(3,3),0)
stages["gaussian blur"] = blur

edges = cv.Canny(blur,50,150)
stages["canny edge detection"] = edges

# fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# for ax, (title, stage_img) in zip(axes.flat, stages.items()):
#     ax.imshow(stage_img, cmap="gray")
#     ax.set_title(title)
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

def preprocess_images(images):
    eq = cv.equalizeHist(images)
    blur = cv.GaussianBlur(eq,(3,3),0)
    edges = cv.Canny(blur,50,150)
    features = edges.flatten()/255.0
    return features

# num_train = 10000
# num_test = 2000

# x_train_sub = np.array([preprocess_images(img) for img in x_train[:num_train]])
# x_test_sub = np.array([preprocess_images(img) for img in x_test[:num_test]])

# y_train_sub = y_train[:num_train]
# y_test = y_test[:num_test]

x_train = np.array([preprocess_images(img) for img in x_train])
x_test = np.array([preprocess_images(img) for img in x_test])

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    batch_size=32,
    verbose = 2
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc," Test loss:", test_loss)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
