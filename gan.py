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
# This file builds/trains the gan
################################################
import config_env as ce

import csv
import chess
import data
import numpy as np
import os
from os.path import basename
from zipfile import ZipFile

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import Model

# Returns the Generator.  Takes in value load (boolean).  If true, 
# the method will load a saved generator. If false, it will build
# a new generator and override any model saved in the default location.
def policy_generator(load):
    # Loads model if true
    if load:
        model = tf.keras.models.load_model(ce.GEN_MODEL_SAVE, compile = False)
        return model
    # Defines input for generator as 8x8x12 model
    input_brd = Input(shape=(8,8,12))
    # Convolutional layer #1
    x = Conv2D(128,(3,3), strides = (1,1), padding = 'same')(input_brd)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #2
    x = Conv2D(128,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #3
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #4
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #5
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #6
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #7
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #8
    x = Conv2D(7,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Builds model
    model = Model([input_brd], [input_brd, x])
    # Outputs a png of the model
    keras.utils.plot_model(model, to_file="gen.png", show_shapes=True)
    # Returns model. Does not need to be compiled because it is compiled
    # as part of the GAN
    return model

# Returns the Discriminator.  Takes in value load (boolean).  If true, 
# the method will load a saved discriminator. If false, it will build
# a new discriminator and override any model saved in the default location.
def policy_discriminator(load):
    # Loads and compiles saved model if true
    if load:
        model = tf.keras.models.load_model(ce.DISC_MODEL_SAVE)
        opt = Adam(learning_rate=ce.LEARNING_RATE)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        return model
    # Defines input for discriminator as 8x8x12 representation of a board and
    # 8x8x7 policy
    input_brd = Input(shape=(8,8,12))
    input_policy = Input(shape=(8,8,7))

    # Concatenate the inputs
    c = Concatenate()([input_brd, input_policy])
    # Convolutional layer #1
    x = Conv2D(128,(3,3), strides = (1,1), padding = 'same')(c)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #2
    x = Conv2D(128,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #3
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #4
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #5
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #6
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #7
    x = Conv2D(64,(3,3), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Convolutional layer #8
    x = Conv2D(7,(1,1), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # Flattens all inputs
    x = Flatten()(x)
    # Creates a fully connected layer from 448 nodes to 16
    x = Dense(64)(x)
    # Creates a fully connected layer from 16 nodes to 1
    x = Dense(1)(x)
    # Builds model
    model = Model([input_brd, input_policy], x)
    # Builds optimizer
    opt = Adam(learning_rate=ce.LEARNING_RATE)
    # Comiles model
    model.compile(loss="binary_crossentropy", optimizer=opt)
    # Outputs png file of discriminator
    keras.utils.plot_model(model, to_file="disc.png", show_shapes=True)
    return model

# Returns the gan. Takes the discriminator and generator as inputs.
def policy_gan(gen, disc):
    # Defines the input to the GAN as an 8x8x12 representation of the board.
    input_brd = Input(shape=(8,8,12))
    # Sets the discriminator to false so it doesn't update during generator training
    disc.trainable=False
    # Connects input to generator
    x = gen(input_brd)
    # Connects generator output to discriminator
    x = disc(x)
    # Builds the model
    model = Model([input_brd], x)
    # Defines the optimizer
    opt = Adam(learning_rate=ce.LEARNING_RATE)
    # Compiles the model
    model.compile(loss="binary_crossentropy", optimizer=opt)

    return model

# Customized training routine for the GAN. This is necessary due to the size of the training set and
# limited resources.
def train():
    # Get the data from the csv file
    chess_dict = data._read_csv()
    # Gets the generator
    gen = policy_generator(False)
    # Gets the compiled discriminator
    disc = policy_discriminator(False)
    # Gets the compiled GAN
    gan = policy_gan(gen, disc)
    # Training to run for the defined number of epochs
    for e in range (ce.EPOCHS):
        # Training will be on the discriminator first, so training needs to be set to true.
        disc = _set_disc_trainable(e,disc, True)
        # Loss is tracked over an epoch, so data is reset at start of each epoch.
        d1_loss = 0
        d2_loss = 0
        g_loss = 0
        # Traing batch loop for discriminator
        for i in range(ce.ELEMENTS//ce.BATCH_SIZE):   
            # Define the board representation and a policy matrix based
            # on the player's actual data
            boards, pols = data.generate_training_batch(chess_dict)
            # Update loss based on training with real data
            d1_loss += train_disc(disc, boards, pols)
            # Updates loss based on training with generated data
            d2_loss += train_disc_on_gen(disc, gen, boards)
        # Sets discriminator training to False for 
        disc = _set_disc_trainable(e,disc, False)
        # Recompiles GAN
        opt = Adam(learning_rate=(0.99 ** e)*ce.LEARNING_RATE)
        gan.compile(loss="binary_crossentropy", optimizer=opt)
        # Training batch loop for generator
        for i in range(ce.ELEMENTS//ce.BATCH_SIZE):
            # Get the board representations
            boards, pols = data.generate_training_batch(chess_dict)
            # Update generator loss based on training generator
            x = train_gen(disc, gan, boards)
            print(x)
            g_loss += x

        # Save the generator and discriminator    
        disc.save(ce.DISC_MODEL_SAVE)
        gen.save(ce.GEN_MODEL_SAVE)
        # Every 10th epoch, zip the folder structure and save in case
        # training zeros out
        if e%10 == 0:
            with ZipFile('gen_save'+str(e)+".zip",'w') as zipObj:
                for folderName, subfolders, filenames in os.walk("./gen_save"):
                    for filename in filenames:
                        filePath = os.path.join(folderName, filename)
                        zipObj.write(filePath, basename(filePath))
            with ZipFile('disc_save'+str(e)+".zip",'w') as zipObj:
                for folderName, subfolders, filenames in os.walk("./disc_save"):
                    for filename in filenames:
                        filePath = os.path.join(folderName, filename)
                        zipObj.write(filePath, basename(filePath))
        save_loss(e, d1_loss, d2_loss, g_loss)
    return

# Sets whether the discriminator is trainable.
def _set_disc_trainable(epoch, model, train):
    model.trainable = train
    # reduces the learning rate after each epoch
    lr = (0.99 ** epoch)*ce.LEARNING_RATE
    # Builds optimizer
    opt = Adam(learning_rate=lr)
    # Compiles model
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model

# Trains the discriminator on real data and returns the loss
def train_disc(disc, boards, pols):
    return disc.train_on_batch([boards, pols], np.ones((boards.shape[0], 1)))
# Trains the discriminator on generated data and returns loss
def train_disc_on_gen(disc, gen, boards):
    _, pols = gen.predict(boards)
    
    return disc.train_on_batch([boards, pols], np.zeros((boards.shape[0], 1)))

# Trains the generator and returns loss
def train_gen(disc, gan, boards):

    return gan.train_on_batch([boards], np.ones((boards.shape[0], 1)))

# After each epoch, the loss data is stored for the epoch
def save_loss(epoch, d1_loss, d2_loss, g_loss):
    with open('disc_loss.csv','a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([epoch, (d1_loss + d2_loss)/(ce.ELEMENTS//ce.BATCH_SIZE)])

    with open('gan_loss.csv','a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([epoch, (g_loss / (ce.ELEMENTS//ce.BATCH_SIZE))])
    return 