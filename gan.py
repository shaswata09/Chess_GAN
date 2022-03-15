
################################################
# CSE 6633 - AI Project
# Chess AI using GAN and RL
# Team Members:
#    - Keith Strandell
#    - Shaswata Mitra
#    - Ivan Fernandez
#    - David Hertz
#    - Sabin Bhujel
#
################################################

# Project file imports
import config_env as ce

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Softmax # Removed from GAN
from tensorflow.keras import Model
from tensorflow.keras.losses import BinaryCrossentropy

##### Generator Structure - Currently uses 7 layers of convolutional blocks. Used to generate a policy block.
def generator():
    input_brd = Input(shape=(8,8,13))
    input_mvs = Input(shape=(8,8,7))

    c = Concatenate()([input_brd, input_mvs])

    # Covolutional Block 1
    x = Conv2D(128, (3, 3), strides=(1,1), padding='same')(c)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 2
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 3
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 4
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)


    # Covolutional Block 5
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 6
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 7
    x = Conv2D(7, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    model = Model([input_brd, input_mvs], x)

    model.summary()
    keras.utils.plot_model(model, to_file="gen.png", show_shapes=True)
    
    return model

##### Discriminator Structure - Currently uses 7 layers of convolutional blocks and a Dense NN.  Generates single predictor.
def discriminator():
    input_brd = Input(shape=(8,8,13))
    input_mvs = Input(shape=(8,8,7))

    c = Concatenate()([input_brd, input_mvs])

    # Covolutional Block 1
    x = Conv2D(128, (3, 3), strides=(1,1), padding='same')(c)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 2
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 3
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 4
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)


    # Covolutional Block 5
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 6
    x = Conv2D(64, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Covolutional Block 7
    x = Conv2D(7, (3, 3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dense(1)(x)
    
    model = Model([input_brd, input_mvs], x)
    
    model.summary()
    keras.utils.plot_model(model, to_file="disc.png", show_shapes=True)
    
    return model

# Loss model for the discriminator
def disc_loss(cross_entropy, pgn_moves, gen_moves):
    real_loss = cross_entropy(tf.ones_like(pgn_moves), pgn_moves)
    gen_loss = cross_entropy(tf.zeros_like(gen_moves), gen_moves)
    total_loss = real_loss + gen_loss
        
    return total_loss
    
# Loss model for the Generator
def gen_loss(cross_entropy, gen_moves):
        
    return cross_entropy(tf.ones_like(gen_moves), gen_moves)
