import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans
import random as r
from keras.optimizers import Adam


import glob
def create_data(src, mask, label=False, resize=(240,240,155)):
    files = glob.glob(src + mask)
    imgs = []
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        if label:
            img[img == 4] = 1
            img[img != 1] = 0
            img = img.astype('float32')
        else:
            img = (img-img.mean()) / img.std()
        img = trans.resize(img, resize, mode='constant')
        imgs.append(img)
    name = 'y' if label else 'x'
    np.save(name, np.array(imgs)[..., np.newaxis].astype('float32'))
    print('Saved', len(files), 'to', name)


from keras.models import Input, Model
from keras.layers import Conv3D, Concatenate, MaxPooling3D, Reshape
from keras.layers import UpSampling3D, Activation, Permute


def level_block_3d(m, dim, depth, factor, acti):
    if depth > 0:
        n = Conv3D(dim, 3, activation=acti, padding='same')(m)
        m = MaxPooling3D()(n)
        m = level_block_3d(m, int(factor*dim), depth-1, factor, acti)
        m = UpSampling3D()(m)
        m = Concatenate(axis=4)([n, m])
    return Conv3D(dim, 3, activation=acti, padding='same')(m)


def UNet_3D(img_shape, n_out=1, dim=64, depth=4, factor=2, acti='relu'):
    i = Input(shape=img_shape)
    o = level_block_3d(i, dim, depth, factor, acti)
    o = Conv3D(n_out, 1, activation='softmax')(o)
    return Model(inputs=i, outputs=o)


import keras.backend as K


def f1_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def f1_loss(y_true, y_pred):
    return -f1_score(y_true, y_pred)


if __name__ == "__main__":
    create_data('/Users/mas/PycharmProjects/Brats17TrainingData/HGG/', '**/*_t1ce.nii.gz', label=False, resize=(64,64,64))
    create_data('/Users/mas/PycharmProjects/Brats17TrainingData/HGG/', '**/*_seg.nii.gz', label=True, resize=(64,64,64))


    x = np.load('x.npy')
    print('x: ', x.shape)
    y = np.load('y.npy')
    print('y:', y.shape)


    i = int(r.random() * x.shape[0])
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(x[i, int(x.shape[1]/2), :, :, 0])
    plt.subplot(122)
    plt.imshow(y[i, int(y.shape[1]/2), :, :, 0])
    plt.show()
    plt.savefig('foo0.png')

    i = int(r.random() * x.shape[0])
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(x[i, int(x.shape[1]/2), :, :, 0])
    plt.subplot(122)
    plt.imshow(y[i, int(y.shape[1]/2), :, :, 0])
    plt.show()
    plt.savefig('foo1.png')


    i = int(r.random() * x.shape[0])
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(x[i, int(x.shape[1]/2), :, :, 0])
    plt.subplot(122)
    plt.imshow(y[i, int(y.shape[1]/2), :, :, 0])
    plt.show()
    plt.savefig('foo2.png')


    i = int(r.random() * x.shape[0])
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(x[i, int(x.shape[1]/2), :, :, 0])
    plt.subplot(122)
    plt.imshow(y[i, int(y.shape[1]/2), :, :, 0])
    plt.show()
    plt.savefig('foo3.png')


    model = UNet_3D(x.shape[1:], dim=16, factor=1)
    model.load_weights('/Users/mas/PycharmProjects/BRATS_proj/weights.h5')
    model.compile(optimizer=Adam(lr=0.000001), loss=f1_loss)
    model.fit(x, y, validation_split=0.2, epochs=50, batch_size=8)
    model.save_weights('weights.h5')
    pred = model.predict(x[:50])

    num = int(x.shape[1]/2)
    for n in range(3):
        i = int(r.random() * pred.shape[0])
        plt.figure(figsize=(15,10))

        plt.subplot(131)
        plt.title('Input')
        plt.imshow(x[i, num, :, :, 0])

        plt.subplot(132)
        plt.title('Ground Truth')
        plt.imshow(y[i, num, :, :, 0])

        plt.subplot(133)
        plt.title('Prediction')
        plt.imshow(pred[i, num, :, :, 0])

        plt.show()
        plt.savefig('foo4.png')


