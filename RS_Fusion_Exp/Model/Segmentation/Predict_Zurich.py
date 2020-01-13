import os

os.environ['KERAS_BACKEND'] = 'theano'
import Data_util
import argparse
import Models
import P_LoadBatches
import glob
import cv2
import numpy as np
import random
import keras
import Data_util
import gdal

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

parser = argparse.ArgumentParser()

parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--epoch_number", type=int, default=5)
parser.add_argument("--test_images", type=str, default="")
parser.add_argument("--test_labels", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--output_label_path", type=str, default="")
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--n_classes", type=int)

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
label_path = args.test_labels
input_width = args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

from keras.models import Model

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)

print(args.save_weights_path + "." + str(epoch_number))
# m.load_weights(args.save_weights_path + "." + str(epoch_number))

for tim in range(35, 37):
    weight_path = './save_weight/ex1.' + str(tim)
    m.load_weights(weight_path)

    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.adam(lr=0.0001),
              metrics=['accuracy'])

    output_height = m.outputHeight
    output_width = m.outputWidth

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg") \
             + glob.glob(images_path + "*.tiff") + glob.glob(images_path + "*.tif")
    images.sort()
    segmentations = glob.glob(label_path + "*.jpg") + glob.glob(label_path + "*.png") + glob.glob(label_path + "*.jpeg") \
                    + glob.glob(label_path + "*.tiff") + glob.glob(label_path + "*.tif")
    segmentations.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]
    sum = []
    sum1 = []
    geo_list = []
    img_size = args.input_width * args.input_height
    for imgName, labelName in zip(images, segmentations):
        outName = imgName.replace(images_path, args.output_path)
        X, im_proj, o_geotransform = P_LoadBatches.getImageArr(imgName, args.input_width, args.input_height)
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

        # 3通道
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
        # seg_img = cv2.resize(seg_img, (input_width, input_height))

        # 单通道
        seg_img1 = np.zeros((output_height, output_width, 1))
        seg_img2 = np.zeros((1, output_height, output_width))
        for c in range(n_classes):
            seg_img1[:, :, 0] += ((pr[:, :] == c) * c).astype('uint8')
            seg_img2[0, :, :] += ((pr[:, :] == c) * c).astype('uint8')

        outLabelName = imgName.replace(images_path, "rData_Zurich3/c1_predictions/")

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(outLabelName, input_width, input_height, 1, gdal.GDT_Byte)
        dataset.SetGeoTransform(o_geotransform)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        for i in range(1):
            dataset.GetRasterBand(i + 1).WriteArray(seg_img2[i, :, :])
        del dataset
        # seg_img = cv2.resize(seg_img, (input_width, input_height))

        # 读入label图像
        label_img, im_proj1, o_geotransform1 = P_LoadBatches.getImageArr(labelName, args.input_width, args.input_height)
        label_img1 = np.zeros((input_width, input_height, 1))
        label_img1 = label_img[0, :, :]

        count_nb = 0
        count = 0
        count1 = 0
        # 计算精度(排除背景类)
        for j in range(input_height):
            for i in range(input_width):
                if label_img1[i, j] == seg_img2[0, i, j]:
                    count1 = count1 + 1
                if label_img1[i, j] != 0:
                    count_nb = count_nb + 1
                    if label_img1[i, j] == seg_img2[0, i, j]:
                        count = count + 1

        accur = count / count_nb
        accur1 = count1 / img_size  # 未排除背景类

        sum.append(accur)
        sum1.append(accur1)

    sum = np.array(sum)
    sum1 = np.array(sum1)
    ave = np.mean(sum)
    ave1 = np.mean(sum1)
    print("平均精度为：%f" % ave)
    print("平均精度(未排除背景类)为：%f" % ave1)

'''
    # ：三通道label
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (LEGEND[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (LEGEND[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (LEGEND[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
'''

'''
    for m in range(output_height):
        for n in range(output_width):
            c = pr[m, n]
            seg_img[m, n, 0] = (LEGEND[c][0]).astype('uint8')
            seg_img[m, n, 1] = (LEGEND[c][1]).astype('uint8')
            seg_img[m, n, 2] = (LEGEND[c][2]).astype('uint8')


    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * LEGEND[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * LEGEND[c][1]).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * LEGEND[c][2]).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
'''
