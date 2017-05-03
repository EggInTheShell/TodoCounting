import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import glob, os
import pandas as pd

coordspath = 'data/coords.csv'
data = pd.read_csv(coordspath)
print(data)

coord = np.asarray(data.as_matrix())
print(coord.shape)

dot_folder = 'data/TrainSmall'

for i in range(coord.shape[0]):
