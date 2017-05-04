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
train_folder = 'data/TrainSmall2/Train/'
# save_folder = 'H:/KaggleNOAASeaLions/classified_images/'
save_folder = DATA_DIR + 'segmentation_images/'
# 保存
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

# トドの座標データを読み込む
data = pd.read_csv(coordspath)
coord = np.asarray(data.as_matrix())
# print(coord.shape)

# 画像データのリストを読み込む
train_images_list  = glob.glob(train_folder+'*.jpg')
# print(train_images_list)

# 各画像を処理する
blurred_white = 0.0176978006093

for imagepath in train_images_list:
    id = int(os.path.basename(imagepath)[:-4])
    if id in black_id_list:
        print(id, 'is black id!')
        continue
    print('train image id ', id)
    image_pil = Image.open(imagepath)
    image = np.asarray(image_pil)
    coord_of_image = coord[coord[:,0]==id]
    print('number of dot: ', coord_of_image.shape[0])
    label = np.zeros([image.shape[0], image.shape[1], 5])

    for i in range(coord_of_image.shape[0]):
        cls = coord_of_image[i,1]
        x = coord_of_image[i,3]
        y = coord_of_image[i,2]
        label[y, x, cls] = 1

    # ブラーをかける
    for i in range(5):
        label[:,:,i] = gaussian_filter(label[:,:,i], sigma=3) / blurred_white
    label = np.minimum(255, label * 255).astype(np.uint8)
    # 可視化
    label_image = np.sum(label, axis=2)
    print(label_image.shape)
    image = image.astype(np.float64)/255
    image = np.minimum(1, image + label_image[:,:,np.newaxis].astype(np.float64)/255)
    image = (image*255).astype(np.uint8)
    # plt.imshow(image)
    # plt.show()



    savepath = save_folder + str(id) + '.pkl'
    with open(savepath, 'wb') as f: # TODO CSV形式化
        pickle.dump(label, f)