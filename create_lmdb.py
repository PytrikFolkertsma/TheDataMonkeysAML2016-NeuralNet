'''
This script resizes the images to 227x227 pixels, enhances the contrast and stores the data into two lmdb databases (one for training and one for validation)
Original script: https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial/blob/master/code/create_lmdb.py
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_lmdb = 'lmdbAML/train_lmdb'
validation_lmdb = 'lmdbAML/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

train_data = [img.split(' ')[0] for img in open('trainset-overview.txt')]
train_labels = [img.split(' ')[1].strip() for img in open('trainset-overview.txt')]

val_data = [img.split(' ')[0] for img in open('valset-overview.txt')]
val_labels = [img.split(' ')[1].strip() for img in open('valset-overview.txt')]

print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        datum = make_datum(img, int(train_labels[in_idx]))
        in_txn.put('{:0>3d}'.format(in_idx), datum.SerializeToString())
        print '{:0>3d}'.format(in_idx) + ':' + img_path
in_db.close()


print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(val_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        datum = make_datum(img, int(val_labels[in_idx]))
        in_txn.put('{:0>3d}'.format(in_idx), datum.SerializeToString())
        print '{:0>3d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'
