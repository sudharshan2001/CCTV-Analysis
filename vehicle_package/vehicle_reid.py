from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Concatenate, Multiply, Subtract
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import cv2


IMAGE_SIZE = [196, 196]
def build_model():
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name='Images')
    inputs2 = keras.Input(shape=(*IMAGE_SIZE, 3), name='Images2')

    inception_model = tf.keras.applications.resnet50.ResNet50(input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3) , include_top=False)
    for layer in inception_model.layers:
        layer.trainable = True
        
#     pooling = GeM(5)(inception_model.output) 
    pooling = tf.keras.layers.GlobalAveragePooling2D()(inception_model.output)

    flatten = layers.Flatten()(pooling)
    dense_1 = layers.Dense(512, activation='relu')(flatten)
    b_norm_1 = layers.BatchNormalization()(dense_1)
    dense_3 = layers.Dense(256, activation='relu')(b_norm_1)
    dense_3 = layers.BatchNormalization()(dense_3)
    
    backbone = keras.Model(inputs=inception_model.inputs, outputs=dense_3, name='effnet')
    
    x1 = backbone(preprocess_input(inputs))
    x2 = backbone(preprocess_input(inputs2))

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    x = Concatenate(axis=-1)([x4, x3])
    
#     x = Concatenate(axis=1)([x1, x2])
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model([inputs,inputs2], x)
    return model

optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001)

model=build_model()
model.summary()