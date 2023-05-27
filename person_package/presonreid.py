from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Concatenate, Multiply, Subtract
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import torch

IMAGE_SIZE = [224,224]

def build_person_reid_model():
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name='Images')
    inputs2 = keras.Input(shape=(*IMAGE_SIZE, 3), name='Images2')

    bacKbone = tf.keras.applications.ResNet101(weights='imagenet', include_top=False,input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    for layer in bacKbone.layers:
        layer.trainable = False
        
    pooling = tf.keras.layers.GlobalAveragePooling2D()(bacKbone.output)

    flatten = layers.Flatten()(pooling)
    dense_1 = layers.Dense(512, activation='relu')(flatten)
    b_norm_1 = layers.BatchNormalization()(dense_1)
    dense_3 = layers.Dense(512, activation='relu')(b_norm_1)
    dense_3 = layers.BatchNormalization()(dense_3)
    
    backbone = keras.Model(inputs=bacKbone.inputs, outputs=dense_3, name='backbone')
    
    x1 = backbone(preprocess_input(inputs))
    x2 = backbone(preprocess_input(inputs2))

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    x = Concatenate(axis=-1)([x4, x3])
    

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model([inputs,inputs2], x)
    model.load_weights('./model_weights/person_reid2.h5')

    return model

def predict_person_reid(image1, image2, model):
    image1 = cv2.resize(image1, (224,224), interpolation = cv2.INTER_AREA)
    image2 = cv2.resize(image2, (224,224), interpolation = cv2.INTER_AREA)
    
    image1 = cv2.resize(image1, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    image1 =  tf.image.convert_image_dtype(image1, tf.float32)

    probability = model.predict(image1,image2)

    return probability

