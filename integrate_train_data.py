import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join, relpath
import glob, os
from scipy.ndimage.filters import gaussian_filter
import pickle
from settings import *
from data_utils import *


# パスは各環境に合わせて書き換える
# TODO データのパスを指定するファイルを設ける
coordspath = 'data/coords.csv'
# train_folder = 'H:/KaggleNOAASeaLions/Train/'
data_folder = DATA_DIR + 'patches/'
# save_folder = 'H:/KaggleNOAASeaLions/classified_images/'
save_folder = DATA_DIR + 'patches/'
# 保存
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)


data_path_list  = glob.glob(data_folder+'*_traindata.pkl')
print(data_path_list)


images = np.zeros([0, 320, 320, 3], dtype=np.uint8)
labels = np.zeros([0, 320, 320, 5], dtype=np.uint8)
for item in data_path_list:
    with open(item, mode='rb') as f:
        dict = pickle.load(f)
    image = dict['image']
    label = dict['label']
    images = np.r_[images, image]
    labels = np.r_[labels, label]

print(images.shape, labels.shape)

image_sum = np.sum(images, axis=1)
image_sum = np.sum(image_sum, axis=1)
image_sum = np.sum(image_sum, axis=1)
print(image_sum.shape)
reduced_images = np.array(images[image_sum!=0])
reduced_labels = np.array(labels[image_sum!=0])
print(reduced_images.shape)

# 保存
dict = {'image': reduced_images, 'label': reduced_labels}
savepath = save_folder + 'traindata.pkl'
with open(savepath, mode='wb') as f:
    pickle.dump(dict, f)
print('saved: ', savepath)