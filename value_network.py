################################################
# CSE 6633 - AI Project
# Chess AI - GAN files
# Team Members:
#    - Keith Strandell
#    - Shaswata Mitra
#    - Ivan Fernandez
#    - David Hertz
#    - Sabin Bhujel
#
# This file builds and trains the value network
################################################
import data
import config_env as ce

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Model

import random
# Returns the value network.  Takes in value load (boolean).  If true, 
# the method will load a saved value network. If false, it will build
# a new value network and override any model saved in the default location.
def v_net(load):
    if load:
        model = tf.keras.models.load_model(ce.VAL_MODEL_SAVE)
        opt = Adam(learning_rate=ce.LEARNING_RATE)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        model.summary()
        return model
    input_brd = Input(shape=(8,8,12))

    x = Conv2D(256,(3,3), strides = (1,1), padding = 'same')(input_brd)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dense(32)(x)
    x = Dense(1)(x)

    model = Model(input_brd, x)
    opt = Adam(learning_rate=ce.LEARNING_RATE)
    model.compile(loss="binary_crossentropy", optimizer=opt)

    return model

# Train a value network.  Takes load (boolean) as an argument to determine 
# whether or not it is loading an existing network to start from.
def train(load):
    # Gets the model to train
    model = v_net(load)
    # Trains for a number of epochs
    for e in range(ce.EPOCH):
        # Read the files
        temp = data._read_csv(ce.FILE_VAL)
        fen_dict = dict(random.sample(temp.items(), ce.VAL_ELEMENTS))
        del(temp)
        loss = 0
        for i in range(ce.VAL_ELEMENTS//ce.BATCH_SIZE): 
            batch,vals = data.generate_val_training_batch(fen_dict)
            loss += model.train_on_batch(batch, vals)
            if i % 25 == 1:
                print(f"{e}-{i}: {loss/i}")
            if i % 500 == 0:
                model.save(ce.VAL_MODEL_SAVE)

    return

