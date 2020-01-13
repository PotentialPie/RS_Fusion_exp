import os
import Data_util
from Data_util import LEGEND
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gdal
import glob
import P_LoadBatches

for iteration in range(20):

    iteration = iteration + 1
    print("The iteration is %d" % iteration)

    # 参考图像路径
    ref_img_path = './Zurich_dataset_v1.0/images_tif/zh' + str(iteration) + '.tif'
    # 待拼接图像路径
    input_img_path = "FC_Exp/zh" + str(iteration) + "/FCN_Results/predictions/"  # 三通道图
    # 待拼接图像路径
    input_c1img_path = "FC_Exp/zh" + str(iteration) + "/FCN_Results/c1predictions/"  # 单通道图
    # 输出图像路径
    out_img_path = "FC_Exp/zh" + str(iteration) + "/Mosaic/M_zh" + str(iteration) + '.tiff'
    # 输出图像路径
    out_c1img_path = "FC_Exp/zh" + str(iteration) + "/c1Mosaic/c1M_zh" + str(iteration) + '.tiff'

    # 读入参考图片
    dataset = gdal.Open(ref_img_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    im_proj = dataset.GetProjection()  # 地图投影信息
    geotransform = dataset.GetGeoTransform()
    del dataset

    # --------------------------------------拼接三通道图像------------------------------------------------

    # 读入带拼接图像路径
    in_images = glob.glob(input_img_path + "*.tiff") + glob.glob(input_img_path + "*.tif")

    # 实现sort排序
    tmp = in_images[0].split('_')[0] + '_' + in_images[0].split('_')[1] + '_' + in_images[0].split('_')[2]
    for i in range(len(in_images)):
        in_images[i] = in_images[i].split('_')[3].split('.')
        in_images[i][0] = int(in_images[i][0])
    in_images.sort()
    for i in range(len(in_images)):
        in_images[i][0] = str(in_images[i][0])
        in_images[i] = tmp + '_' + in_images[i][0] + '.' + in_images[i][1]

    # 带拼接图像数量
    num = len(in_images)
    # 写入图像
    bands = 3
    temp = np.zeros((bands, width, height))
    re_width = 256
    re_height = 256
    Xt0 = geotransform[0]
    Yt0 = geotransform[3]

    for i in range(num):
        re_img, im_proj, o_geotransform = P_LoadBatches.getImageArr(in_images[i], re_width, re_height)
        XRes = o_geotransform[1]  # width
        YRes = o_geotransform[5]  # height
        X0 = o_geotransform[0]
        Y0 = o_geotransform[3]
        for rows in range(re_height):
            Ya = Y0 + YRes * rows
            for cols in range(re_width):
                Xa = X0 + XRes * cols
                tx = round((Xa - Xt0) / XRes)
                ty = round((Ya - Yt0) / YRes)
                if tx >= width:
                    tx = width - 1
                if ty >= height:
                    ty = height - 1
                temp[:, tx, ty] = re_img[:, rows, cols]
    #    a = np.rollaxis(temp, 0, 3)
    #    plt.imshow(a)
    #    plt.show()

    # 输出图片其他参数
    datatype = gdal.GDT_Byte
    temp = np.rollaxis(temp, 2, 1)

    # 保存图片
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(out_img_path, width, height, bands, datatype)
    dataset.SetGeoTransform(geotransform)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    for i in range(bands):
        dataset.GetRasterBand(i + 1).WriteArray(temp[i])
    del dataset

    # --------------------------------------拼接单通道图像------------------------------------------------

    # 读入带拼接图像路径
    in_images = glob.glob(input_c1img_path + "*.tiff") + glob.glob(input_c1img_path + "*.tif")

    # 实现sort排序
    tmp = in_images[0].split('_')[0] + '_' + in_images[0].split('_')[1] + '_' + in_images[0].split('_')[2]
    for i in range(len(in_images)):
        in_images[i] = in_images[i].split('_')[3].split('.')
        in_images[i][0] = int(in_images[i][0])
    in_images.sort()
    for i in range(len(in_images)):
        in_images[i][0] = str(in_images[i][0])
        in_images[i] = tmp + '_' + in_images[i][0] + '.' + in_images[i][1]

    # 带拼接图像数量
    num = len(in_images)
    # 写入图像
    bands = 1
    temp = np.zeros((bands, width, height))
    re_width = 256
    re_height = 256
    Xt0 = geotransform[0]
    Yt0 = geotransform[3]

    for i in range(num):
        re_img, im_proj, o_geotransform = P_LoadBatches.getImageArr(in_images[i], re_width, re_height)
        XRes = o_geotransform[1]  # width
        YRes = o_geotransform[5]  # height
        X0 = o_geotransform[0]
        Y0 = o_geotransform[3]
        for rows in range(re_height):
            Ya = Y0 + YRes * rows
            for cols in range(re_width):
                Xa = X0 + XRes * cols
                tx = round((Xa - Xt0) / XRes)
                ty = round((Ya - Yt0) / YRes)
                if tx >= width:
                    tx = width - 1
                if ty >= height:
                    ty = height - 1
                temp[:, tx, ty] = re_img[:, rows, cols]

    # 输出图片其他参数
    datatype = gdal.GDT_Byte
    temp = np.rollaxis(temp, 2, 1)

    # 保存图片
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(out_c1img_path, width, height, bands, datatype)
    dataset.SetGeoTransform(geotransform)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    for i in range(bands):
        dataset.GetRasterBand(i + 1).WriteArray(temp[i])
    del dataset

print("Finish!!!")
