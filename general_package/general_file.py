import tensorflow as tf
IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 20
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, GRU, Dense,Activation
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
CLASSES_LIST = ['RoadAccidents','Explosion','Fighting'] #os.listdir('/kaggle/input/anomaly/anomaly images') 
rnn_size = 32

def get_model():
    model = Sequential()
    model.add(LSTM(rnn_size, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
    # model.add(LSTM(64))
    # model.add(GRU(512))
    # model.add(Activation('relu'))
    # model.add(Dense(32))
    # model.add(Activation('relu'))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dense(128))
    # model.add(Activation('sigmoid'))
    model.add(Dense(len(CLASSES_LIST)))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001),
                metrics=['categorical_accuracy', tf.keras.metrics.Precision()])
    model.load_weights('./model_weights/general_sur.h5')
    return model


def build_feature_extractor():
    feature_extractor = tf.keras.applications.ResNet101(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMAGE_HEIGHT, IMAGE_HEIGHT, 3),
    )
    preprocess_input = tf.keras.applications.resnet.preprocess_input

    inputs = tf.keras.Input((IMAGE_HEIGHT, IMAGE_HEIGHT, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

# feature_extractor = build_feature_extractor()