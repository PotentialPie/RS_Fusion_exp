import os

os.environ['KERAS_BACKEND'] = 'theano'
import argparse
import Models
import P_LoadBatches
import glob
import numpy as np
import random
import keras
import gdal
import matplotlib.pyplot as plt
from keras.models import Model

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

# 参数
n_classes = 9
model_name = 'fcn8'
images_path = "Data_Zurich11/Image_Train/"
label_path = "Data_Zurich11/Label_Train/"
weight_path = "Opt_weights/ex"
input_width = 256
input_height = 256
output_height = 256
output_width = 256
epoch_number = 60
modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}

for tim in range(19, 20):
    tim = tim + 1

    # 预测图像路径
    print('这是第%d张图'%tim)
    out_image_path = "FC_Exp/zh" + str(tim) + "/FCN_Results/predictions/"  # 三通道图
    out_c1image_path = "FC_Exp/zh" + str(tim) + "/FCN_Results/c1predictions/"  # 单通道图
    # 倒数第一层特征路径
    out_L1feature_path = "FC_Exp/zh" + str(tim) + "/Last1_Feature/"
    # 倒数第二层特征路径
    out_L2feature_path = "FC_Exp/zh" + str(tim) + "/Last2_Feature/"

    print('加载模型')
    modelFN = modelFns[model_name]

    m = modelFN(n_classes, input_height=input_height, input_width=input_width)

    weight_path1 = weight_path + str(tim) + '.*'
    opt_weight_path = glob.glob(weight_path1)
    m.load_weights(opt_weight_path[0])

    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.adam(lr=0.0001),
              metrics=['accuracy'])

    # 输出最后一层特征
    last_layer_model = Model(inputs=m.inputs, outputs=m.get_layer(name='conv2d_transpose_3').output)
    # 输出倒数第二层特征
    last2_layer_model = Model(inputs=m.inputs, outputs=m.get_layer(name='add_2').output)

    # ---------------------------输出测试集预测图片----------------------------------------
    # 从1-20里面选出测试集来
    images_path1 = images_path + 'zh' + str(tim) + '_'
    label_path1 = label_path + 'zh' + str(tim) + '_'

    print("The Testing image Path:")
    print(images_path1 + "*.tiff")
    print(label_path1 + "*.tiff")

    images = glob.glob(images_path1 + "*.tiff") + glob.glob(images_path1 + "*.tif")
    images.sort()
    segmentations = glob.glob(label_path1 + "*.tiff") + glob.glob(label_path1 + "*.tif")
    segmentations.sort()

    sum = []
    sum1 = []
    geo_list = []
    img_size = input_width * input_height
    for imgName, labelName in zip(images, segmentations):
        outName = imgName.replace(images_path, out_image_path)
        X, im_proj, o_geotransform = P_LoadBatches.getImageArr(imgName, input_width, input_height)
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

        # 输出3通道测试集预测图
        seg_img = np.zeros((3, output_width, output_height))
        for c in range(n_classes):
            seg_img[0, :, :] += ((pr[:, :] == c) * (LEGEND[c][0])).astype('uint8')
            seg_img[1, :, :] += ((pr[:, :] == c) * (LEGEND[c][1])).astype('uint8')
            seg_img[2, :, :] += ((pr[:, :] == c) * (LEGEND[c][2])).astype('uint8')

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(outName, input_width, input_height, 3, gdal.GDT_Byte)
        dataset.SetGeoTransform(o_geotransform)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        for i in range(3):
            dataset.GetRasterBand(i + 1).WriteArray(seg_img[i, :, :])
        del dataset

        # 输出单通道测试集预测图
        seg_img2 = np.zeros((1, output_height, output_width))
        for c in range(n_classes):
            seg_img2[0, :, :] += ((pr[:, :] == c) * c).astype('uint8')

        outLabelName = imgName.replace(images_path, out_c1image_path)

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(outLabelName, input_width, input_height, 1, gdal.GDT_Byte)
        dataset.SetGeoTransform(o_geotransform)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        for i in range(1):
            dataset.GetRasterBand(i + 1).WriteArray(seg_img2[i, :, :])
        del dataset

    # ---------------------------输出所有图片最后两层特征----------------------------------------
    # 读入所有图片
    images = glob.glob(images_path + "*.tiff") + glob.glob(images_path + "*.tif")
    images.sort()
    segmentations = glob.glob(label_path + "*.tiff") + glob.glob(label_path + "*.tif")
    segmentations.sort()
    sum = []
    sum1 = []
    geo_list = []
    img_size = input_width * input_height
    for imgName, labelName in zip(images, segmentations):

        X, im_proj, o_geotransform = P_LoadBatches.getImageArr(imgName, input_width, input_height)

        # 输出最后一层特征
        outName = imgName.replace(images_path, out_L1feature_path)
        outName_l1 = outName.replace('tiff', 'npy')
        npy_last_layer = last_layer_model.predict(np.array([X]))[0]
        np.save(outName_l1, npy_last_layer)

        # 输出倒数第二层特征
        outName = imgName.replace(images_path, out_L2feature_path)
        outName_l2 = outName.replace('tiff', 'npy')
        npy_last2_layer = last2_layer_model.predict(np.array([X]))[0]
        np.save(outName_l2, npy_last2_layer)

    del m