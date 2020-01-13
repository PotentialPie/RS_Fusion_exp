import os
import Data_util
from Data_util import LEGEND
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gdal

# 图片和语义分割图的主目录
base_data_path = './Zurich_dataset_v1.0'
IMG_path = os.path.join(base_data_path, 'images_tif')
LABEL_path = os.path.join(base_data_path, 'groundtruth')

# 图片和语义分割图的地址
test_img_path = './Zurich_dataset_v1.0/images_tif/zh16.tif'
test_label_path = './Zurich_dataset_v1.0/groundtruth/zh16_GT.tif'

# 读取语义分割图，以label的方式也就是(height,width)
label_test = Data_util.ReadImgByPath(test_label_path, is_label=True)  # wwt:是label形式的话将三通道转换成单通道的label形式(像素范围0-8)
# 读取语义分割图，以非label的方式也就是(3, height,width),与类别的编码关系见Data_util的LEGEND
label_test_n = Data_util.ReadImgByPath(test_label_path, is_label=False)
# 读取图像（wwt:实际遥感图像）
image_test = Data_util.ReadImgByPath(test_img_path)

# 显示遥感图像，4通道的图像这里只能显示三通道（height width channel）
print(image_test[0:3, :, :].shape)
# 调整前是（C H W）调整后是（H W C）
img1 = np.rollaxis(image_test[0:3, :, :],0,3)
print(img1.shape)
plt.imshow(img1)
plt.show()

print(label_test.shape)
print(label_test_n.shape)
print(image_test.shape)

# 计数工具，用于统计一个语义分割图（label形式）的类别分布
print(Data_util.Count(label_test))
# 对语义分割图做one hot编码，把(height, width)的label变成(height, width, num_of_label)的样子
one_hot_label = Data_util.one_hot_encode(label_test, 9)
'''
    255    255    255;  % Background
      0      0      0;  % Roads
    100    100    100;  % Buildings
      0    125      0;  % Trees
      0    255      0;  % Grass
    150     80      0;  % Bare Soil
      0      0    150;  % Water
    255    255      0;  % Railways
    150    150    255]; % Swimming Pools 
'''
# 这边随机找几个像素点进行验证,对应方式是上边注释
test_m = 100
test_n = 198
print(one_hot_label[test_m, test_n, :])
print(label_test[test_m, test_n])
print(label_test_n[:, test_m, test_n])

# 对图像进行切割做数据增强，size是切割后图片的大小，stride是切割时slide windows的步长
image_list, label_list = Data_util.Crop(image_test, label_test, size=224, stride=100)

# cv2.imshow(label_test)

plt.imshow(label_test)
plt.show()
