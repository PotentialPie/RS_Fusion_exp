# -------------------------------------------------------------
# Author:wwt117@163.com
# Descri: Output the optimal weights model of each images
# --------------------------------------------------------------

import os

os.environ['KERAS_BACKEND'] = 'theano'
import P_LoadBatches
import glob
import numpy as np
import shutil
import keras
import Models
import D_LoadBatches

LEGEND = [[255, 255, 255],
          [0, 0, 0],
          [100, 100, 100],
          [0, 125, 0],
          [0, 255, 0],
          [150, 80, 0],
          [0, 0, 150],
          [255, 255, 0],
          [150, 150, 255]]

# 训练参数
train_batch_size = 4
val_batch_size = 2
n_classes = 9
input_height = 256
input_width = 256
validate = 0  # -------不使用验证集----------
save_weights_path = 'Weights/ex1'
epochs = 50
model_name = 'fcn8'
train_images_path = "Data_Zurich11/Image_Train/zh"
train_segs_path = "Data_Zurich11/Label_Train/zh"
val_images_path = ''
val_segs_path = ''
modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}

# 测试参数
images_path = "Data_Zurich11/Image_Train/zh"
label_path = "Data_Zurich11/Label_Train/zh"
input_weight_path = 'Weights/ex1.'
opt_weight_path = 'Opt_weights/ex'

print("---------------------------------------------")
print("Begin Processinging:")
print("---------------------------------------------\n")
for tim in range(9, 20):
    tim = tim + 1  # tim:20张图像g
    print('The Training Image is %d' % tim)

    modelFN = modelFns[model_name]

    m = modelFN(n_classes, input_height=input_height, input_width=input_width)
    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.adam(lr=0.0001),
              metrics=['accuracy'])

    # 使用pre-trained
    # m.load_weights('weights3/exf1.9')

    print("Model output shape", m.output_shape)

    output_height = m.outputHeight
    output_width = m.outputWidth

    G = D_LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                                 input_height, input_width, output_height, output_width, tim)

    if validate:
        G2 = D_LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes,
                                                      input_height,
                                                      input_width, output_height, output_width, tim)

    if not validate:
        for ep in range(epochs):
            print("The epochs is %d" % ep)
            m.fit_generator(G, 512, epochs=1)
            m.save_weights(save_weights_path + "." + str(ep))
            print('The output weights path:\t')
            print(save_weights_path + "." + str(ep))
    else:
        for ep in range(epochs):
            print("The epochs is %d" % ep)
            m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
            m.save_weights(save_weights_path + "." + str(ep))
            print('The output weights path:\t')
            print(save_weights_path + "." + str(ep))
    print("The %d trained image is Processed!!!" % tim)

    # ---------------------------------------------------------我是分割线---------------------------------------------------

    print("---------------------------------------------")
    print('The Predicting Image is %d' % tim)
    print("---------------------------------------------\n")

    m = modelFN(n_classes, input_height=input_height, input_width=input_width)

    # 从1-20里面选出测试集来
    images_path1 = images_path + str(tim) + '_'
    label_path1 = label_path + str(tim) + '_'

    print("The Testing image Path:")
    print(images_path1 + "*.tiff")
    print(label_path1 + "*.tiff")

    images = glob.glob(images_path1 + "*.tiff") + glob.glob(images_path1 + "*.tif")
    images.sort()
    segmentations = glob.glob(label_path1 + "*.tiff") + glob.glob(label_path1 + "*.tif")
    segmentations.sort()

    # 第n次最优模型
    max_weight_num = 0
    max_ave = 0
    for t_w in range(epochs):
        weight_path = input_weight_path + str(t_w)  # t_w: 60个权重
        m.load_weights(weight_path)

        m.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.adam(lr=0.0001),
                  metrics=['accuracy'])
        sum = []
        sum1 = []
        geo_list = []
        img_size = input_width * input_height
        for imgName, labelName in zip(images, segmentations):
            X, im_proj, o_geotransform = P_LoadBatches.getImageArr(imgName, input_width, input_height)
            pr = m.predict(np.array([X]))[0]
            pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

            # 单通道
            seg_img2 = np.zeros((1, output_height, output_width))
            for c in range(n_classes):
                seg_img2[0, :, :] += ((pr[:, :] == c) * c).astype('uint8')

            # 读入label图像
            label_img, im_proj1, o_geotransform1 = P_LoadBatches.getImageArr(labelName, input_width, input_height)
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

        if ave > max_ave:
            max_ave = ave
            max_weight_num = t_w

    print("The %d predicted image is Processed!!!" % tim)

    # 输出最优模型
    aim_weight_path = input_weight_path + str(max_weight_num)
    opt_weight_path1 = opt_weight_path + str(tim) + '.' + str(max_weight_num)
    shutil.copy(aim_weight_path, opt_weight_path1)
    print("The optimal weights:")
    print(opt_weight_path1)

print("---------------------------------------------")
print("Finish Processinging:")
print("---------------------------------------------\n")
