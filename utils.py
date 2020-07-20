import tensorflow as tf
import numpy as np
import random
import math
import h5py
import os
from PIL import Image


''' Invert Color Channel '''
def rgb2ycbcr(img):
    cvt_img = img.convert('YCbCr') 
    return cvt_img

def ycbcr2rgb(img):
    cvt_img = img.convert('RGB')
    return cvt_img

'''  '''
def imsave(img, filename):
    img.save(filename, format='BMP')

def imread(img_dir):
    return Image.open(img_dir)

def time_calculate(sec):
    s = sec % 60
    m = sec // 60
    h = m // 60
    m = m % 60
    return h, m, s

def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

''' For load training datasets '''
def load_train(color_space='ycbcr'):
    color_space = color_space.lower()
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'Train', 'DIV2K_train_HR')
    images = []
    
    for img in os.listdir(DATA_DIR):
        image = imread(os.path.join(DATA_DIR, img))
        if color_space == 'ycbcr':
            image = rgb2ycbcr(image)
        images.append(image)
    return images[:500]

''' For load testing datasets '''
def load_set5(color_space='ycbcr'):
    color_space = color_space.lower()
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set5')
    images = []
    
    for img in os.listdir(DATA_DIR):
        if img[-3:] == 'bmp':
            image = imread(os.path.join(DATA_DIR, img))
            if color_space == 'ycbcr':
                image = rgb2ycbcr(image)
            images.append(image)
    return images

def load_set14(color_space='ycbcr'):
    color_space = color_space.lower()
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set14')
    images = []
    
    for img in os.listdir(DATA_DIR):
        if img[-3:] == 'bmp':
            image = imread(os.path.join(DATA_DIR, img))
            if color_space == 'ycbcr':
                image = rgb2ycbcr(image)
            images.append(image)
    return images

def load_b100(color_space='ycbcr'):
    color_space = color_space.lower()
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'B100')
    images = []
    
    for img in os.listdir(DATA_DIR):
        if img[-3:] == 'bmp':
            image = imread(os.path.join(DATA_DIR, img))
            if color_space == 'ycbcr':
                image = rgb2ycbcr(image)
            images.append(image)
    return images

def load_urban100(color_space='ycbcr'):
    color_space = color_space.lower()
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Urban100')
    images = []
    
    for img in os.listdir(DATA_DIR):
        if img[-3:] == 'bmp':
            image = imread(os.path.join(DATA_DIR, img))
            if color_space == 'ycbcr':
                image = rgb2ycbcr(image)
            images.append(image)
    return images

''' preprocessing '''
def preprocessing(x):
    x = to_array(x)
    x = x[:,:,:,0:1] # get only Y channel
    x = normalize(x)
    x = shuffle(x)
    return x

def prepare_patches(x, patch_size=100, stride=25):
    patches = []
    for idx in range(len(x)):
        img = x[idx]
        h, w = img.size
        h_cnt = (h - patch_size) // stride + 1
        w_cnt = (w - patch_size) // stride + 1    

        for i in range(h_cnt):
            for j in range(w_cnt):
                img_crop = img.crop((i*stride, j*stride, patch_size+i*stride, patch_size+j*stride))
                patches.append(img_crop)

    return patches

def bicubic_downsampling(x, scale=4):
    if not isinstance(x, list):
        x = [x]
    
    bicubic = []
    for idx in range(len(x)):
        img = x[idx]
        h, w = img.size
        output_shape = (h//scale, w//scale)
        img = img.resize(output_shape, Image.BICUBIC)
        bicubic.append(img)
    
    return bicubic

def bicubic_upsampling(x, scale=4):
    if not isinstance(x, list):
        x = [x]
    
    bicubic = []
    for idx in range(len(x)):
        img = x[idx]
        h, w = img.size
        output_shape = (h*scale, w*scale)
        img = img.resize(output_shape, Image.BICUBIC)
        bicubic.append(img)
    
    return bicubic

def to_array(x):
    data = []
    for idx in range(len(x)):
        img = x[idx]
        data.append(np.array(img))
    return np.array(data)

def shuffle(x):
    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    return x

def normalize(x):
    return np.array(x) / 255.

def denormalize(x):
    x *= 255.
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return np.array(x)

def mod_crop(x, scale):
    h, w = x.size
    cropped = x.crop((0, 0, h - h%scale, w - w%scale))
    return cropped

if __name__ == '__main__':
    x = load_train()
    print(len(create_sub_patches(x[:500], 100, 100)))