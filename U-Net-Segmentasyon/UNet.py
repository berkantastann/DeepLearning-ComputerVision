import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2 as cv


def load_dataset(root,img_size=(128,128)):
    images,masks = [],[]
    
    for tile in sorted(os.listdir(root)):
        img_dir = os.path.join(root,tile,'images')
        mask_dir = os.path.join(root,tile,'masks')
        if not os.path.isdir(img_dir): continue
        
        for f in os.listdir(img_dir):
            if not f.endswith(".jpg"): continue
            img_path = os.path.join(img_dir,f)
            mask_path = os.path.join(mask_dir,os.path.splitext(f)[0]+'.png')
            if not os.path.exists(mask_path): continue
        
            img = cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2RGB)
            img = cv.resize(img,img_size)
            
            mask = cv.imread(mask_path,cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask,img_size)
            
            mask = np.expand_dims(mask,axis=-1)/255.0
            
            images.append(img)
            masks.append(mask)
            
    return np.array(images,dtype="float32"),np.array(masks,dtype="float32")

x,y = load_dataset("aerial_dataset",img_size=(128,128))
print("Images shape:",x.shape)
print("Masks shape:",y.shape)

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42)
print("Train images:",x_train.shape)
print("Validation images:",x_val.shape)

def unet_model(input_size=(128,128,3)):

    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(16,(3,3),activation='relu',padding='same')(inputs)
    c1 = layers.Conv2D(16,(3,3),activation='relu',padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)
    
    c2 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(p1)
    c2 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)
    
    c3 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(p2)
    c3 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)
    
    c4 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(p3)
    c4 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(256,(3,3),activation='relu',padding='same')(p4)
    c5 = layers.Conv2D(256,(3,3),activation='relu',padding='same')(c5)
    
    # Decoder
    
    u6 = layers.Conv2DTranspose(128,2,strides=(2,2),padding='same')(c5)
    u6 = layers.concatenate([u6,c4])
    c6 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(u6)
    c6 = layers.Conv2D(128,(3,3),activation='relu',padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(64,2,strides=(2,2),padding='same')(c6)
    u7 = layers.concatenate([u7,c3])
    c7 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(u7)
    c7 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(32,2,strides=(2,2),padding='same')(c7)
    u8 = layers.concatenate([u8,c2])
    c8 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(u8)
    c8 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(16,2,strides=(2,2),padding='same')(c8)
    u9 = layers.concatenate([u9,c1])
    c9 = layers.Conv2D(16,(3,3),activation='relu',padding='same')(u9)
    c9 = layers.Conv2D(16,(3,3),activation='relu',padding='same')(c9)
    
    outputs = layers.Conv2D(1,1,activation='sigmoid')(c9)
    
    return keras.Model(inputs,outputs)

unet_model = unet_model()

unet_model.compile(optimizer='adam',loss='binary_crossentropy')

callbacks = [
    keras.callbacks.ModelCheckpoint("unet_best_model.h5",save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),
]

history = unet_model.fit(
    x_train,y_train,
    validation_data=(x_val,y_val),
    epochs=50 ,
    batch_size=16,
    callbacks=callbacks
)

plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend()
plt.show()

def show_prediction(model,idx):
    img = x_val[idx]
    mask = y_val[idx]
    pred_mask = model.predict(np.expand_dims(img,axis=0))[0]
    pred_mask = (pred_mask>0.5).astype(np.float32)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("True Mask")
    plt.imshow(mask.squeeze(),cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask.squeeze(),cmap='gray')
    plt.axis('off')
    plt.show()
    