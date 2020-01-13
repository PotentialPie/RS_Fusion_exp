import os
import Data_util
from Data_util import LEGEND
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gdal
import glob
import Data_util

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
LEGEND = [[255, 255, 255],
          [0, 0, 0],
          [100, 100, 100],
          [0, 125, 0],
          [0, 255, 0],
          [150, 80, 0],
          [0, 0, 150],
          [255, 255, 0],
          [150, 150, 255]]

for iteration in range(20):

    iteration = iteration + 1
    print("The iteration is %d" % iteration)

    test_img_path = "FC_Exp/zh" + str(iteration) + "/c1Mosaic/c1M_zh" + str(iteration) + '.tiff'
    ref_img_path = './Zurich_dataset_v1.0/groundtruth/zh' + str(iteration) + '_GT.tif'

    # 测试图片
    dataset = gdal.Open(test_img_path)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    test_img = [dataset.GetRasterBand(i).ReadAsArray(0, 0, cols, rows) for i in range(1, bands + 1)]
    test_img = np.array(test_img)
    del dataset

    # 参考图片
    dataset1 = gdal.Open(ref_img_path)
    cols1 = dataset1.RasterXSize
    rows1 = dataset1.RasterYSize
    bands1 = dataset1.RasterCount
    ref_img = [dataset1.GetRasterBand(i).ReadAsArray(0, 0, cols1, rows1) for i in range(1, bands1 + 1)]
    ref_img = np.array(ref_img)
    ref_img_c1 = np.zeros((ref_img.shape[1], ref_img.shape[2]), dtype=int)
    # print(Label.shape)
    for i in range(len(LEGEND)):
        for m in range(ref_img.shape[1]):
            for n in range(ref_img.shape[2]):
                if ref_img[0][m][n] == LEGEND[i][0] and ref_img[1][m][n] == LEGEND[i][1] and ref_img[2][m][n] == \
                        LEGEND[i][2]:
                    ref_img_c1[m][n] = i
    del dataset1


    count = 0
    count_nb = 0
    count1 = 0
    # 计算对应相同的点
    for i in range(cols1):
        for j in range(rows1):
            if test_img[0, j, i] == ref_img_c1[j, i]:
                count1 = count1 + 1
            if ref_img_c1[j, i] != 0:
                count_nb = count_nb + 1
                if test_img[0, j, i] == ref_img_c1[j, i]:
                    count = count + 1

    accur = count / count_nb
    accur1 = count1 / (cols1 * rows1)

    print("FCN预测精度为：%f" % accur)
    print("FCN预测精度(未排除背景类)为：%f" % accur1)
