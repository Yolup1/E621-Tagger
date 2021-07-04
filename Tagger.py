# -*- coding: utf-8 -*-

"""
Created on Sun Mar 28 02:17:50 2021

@author: Yolup1
"""

source_dir = 'source/*.png'
IMAGE_SIZE = 512
tags_dir = 'data/Tags.csv';
weights_dir = 'data/Weights.csv'
checkpoint_Best_path = 'results/best/CheckPoint.ckpt'



import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, experimental, GlobalMaxPooling2D
from keras.models import Sequential
from glob import glob

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0] ** (1 - y_true)) * (weights[:,1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),axis = -1)
    return weighted_loss

def sort_dict(indict):
    sorted_dict = sorted(indict,key = indict.get,reverse = True)
    return sorted_dict

weights = np.genfromtxt(weights_dir,delimiter = ',').astype(np.float32)

''' This Block was needed on my previous version of TF, 2.4.0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
'''

remove = [
    'pokémon_(species)','hi_res','absurd_res','pokémon','1:1','<3','2019','2020',
    '2018','2017','2016','2015','animated','widescreen','2014','16:9','3:4','4:3',
    'unknown_artist','2013','alpha_channel','sonic_the_hedgehog_(series)','zootopia',
    '2012','low_res','digimon','digimon_(species)','legendary_pokémon','twilight_sparkle_(mlp)',
    'age_difference','4:5','translated','webm','incest_(lore)','source_filmmaker',
    'judy_hopps','transparent_background','animal_crossing','poképhilia','rainbow_dash_(mlp)',
    'fluttershy_(mlp)','2:3','webcomic_character','gloves_(marking)','crossover',
    'nick_wilde','sibling','2021','3d_animation','2d_animation','5:4','2011','no_sound',
    'video_games','digital_media_(artwork)','english_text','my_little_pony','hasbro',
    'friendship_is_magic','dialogue','lagomorph','young','anthrofied','pony','non-mammal_breasts',
    'fan_character','cub','disney','hybrid','sketch','backsack','girly','url',
    'mammal_humanoid','earth_pony','short_playtime','pegasus','japanese_text',
    'digitigrade','makeup','digital_drawing_(artwork)','alien','arthropod','mustelid',
    'traditional_media_(artwork)','winged_unicorn','bovine','cervid','holidays',
    'low_res','pokémorph','crossgender','eeveelution','plantigrade','perineum',
    'eulipotyphlan','translucent','hedgehog','gloves_(marking)','mouse','nordic_sled_dog',
    'magic','mario_bros','marsupial','monster','sciurid','sound_effects','parent'
]

remove = set(remove)

tag_names = pd.read_csv(tags_dir).drop(remove,axis=1).columns


#
#   Model Definition
#

model = Sequential([
    Input(shape = (None,None,3)),

    experimental.preprocessing.RandomRotation(0.2),

    Conv2D(8,(3,3),activation = 'relu'),
    Conv2D(64,(3,3),activation = 'relu'),

    MaxPooling2D(strides=(2,2)),

    Conv2D(256,(3,3),activation = 'relu'),
    Conv2D(256,(3,3),activation = 'relu'),

    MaxPooling2D(strides = (2,2)),

    Conv2D(512,(3,3),activation = 'relu'),
    Conv2D(512,(3,3),activation = 'relu'),
    Conv2D(512,(3,3),activation = 'relu'),

    GlobalMaxPooling2D(),

    Flatten(),

    Dense(512,activation = 'relu'),
    Dropout(0.2),

    Dense(512,activation = 'relu'),
    Dense(len(tag_names),activation = 'sigmoid')
])

model.load_weights(checkpoint_Best_path)



image = glob(source_dir)


#
#   Prediction
#

j = 0
imgs = dict()
predictions = dict()

for i in image:
    img = tf.io.read_file(i)
    img = tf.image.decode_jpeg(img,channels = 3)
    img = tf.cast(img,tf.float32) / 255
    img = tf.image.resize(img,(IMAGE_SIZE,IMAGE_SIZE),antialias = True,preserve_aspect_ratio = True)
    img_array = tf.expand_dims(img, 0)
    imgs[j] = img
    prediction = model.predict(img_array)
    predictions[j] = prediction
    j += 1

print(predictions[0][0][0],predictions[1][0][0])

testags = dict()
isort = dict()
imax = dict()


#
#   Show tags with >= 65% confidence
#

for p,t in zip(predictions.values(),range(len(image))):
    for i,j in zip(p[0],range(50)):
        if i>0.65:
            testags[tag_names[j]] = i

    isort[t] = sort_dict(testags)
    imax[t] = list(isort[t][:3])

for i in range(len(image)):
    print ('\n\n\n',isort[i])


#
#   Show images with plt
#

for i in range(len(imgs)):
    plt.tight_layout()
    plt.imshow(imgs[i])
    plt.xticks([])
    plt.yticks([])
    plt.title("tags: {}".format(imax[i][:3]))
    plt.show()
