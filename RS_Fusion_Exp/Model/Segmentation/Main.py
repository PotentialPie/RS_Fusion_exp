import os
import Data_util
from Data_util import LEGEND
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gdal
import random
from sklearn import preprocessing


def PatchGenerator1():
    row, column = 256, 256
    image = np.zeros([row, column])
    tmp_size, loop_size = 0, random.randint(3000, 8000)
    print("iter", loop_size)
    action_state = []
    x, y = int(random.random() * 1000) % row, int(random.random() * 1000) % column
    image[x][y] = 1
    tmp_size += 1
    while (tmp_size < loop_size):
        # 随机动作
        action = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        random.shuffle(action)

        ptr = tmp_size
        for i in range(4):
            tmp_x = x + action[i][0]
            tmp_y = y + action[i][1]

            # 满足位置条件
            if 0 <= tmp_x < row and 0 <= tmp_y < column:
                # print('放之前：',tmp_size,':',tmp_x,tmp_y,image[tmp_x][tmp_y])
                if image[tmp_x][tmp_y] != 1:
                    image[tmp_x][tmp_y] = 1
                    tmp_size += 1
                    x, y = tmp_x, tmp_y
                    action_state.append([-action[i][1], -action[i][0]])
        if ptr == tmp_size:
            # 随机走完之后体积都不变
            if len(action_state):
                t = action_state.pop(-1)
                x, y = x + t[0], y + t[1]
            else:
                x, y = int(random.random() * 1000) % row, int(random.random() * 1000) % column

    return image


def PatchGenerator2():
    row, column = 256, 256

    image2 = np.zeros([row, column])
    # 当前面积，总面积

    tmp_size, loop_size = 0, random.randint(10000, 25000)

    state_point = []
    x, y = int(random.random() * 1000) % row, int(random.random() * 1000) % column

    state_point.append([x, y])

    while tmp_size < loop_size:
        # 随机在边界池中取一个
        i = random.randint(0, len(state_point))

        if i >= len(state_point):
            i -= 1

        a, b = state_point[i]
        # print('当前位置',a,b,i)

        # 遍历四周并填充，然后将这些点放入边界池
        action = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for k in range(4):
            tmp_a = a + action[k][0]
            tmp_b = b + action[k][1]
            if 0 <= tmp_a < row and 0 <= tmp_b < column and image2[tmp_a][tmp_b] != 1:
                image2[tmp_a][tmp_b] = 1
                state_point.append([tmp_a, tmp_b])
                tmp_size += 1
                # print('当前体积',tmp_size)
    print("="*10,np.sum(image2))
    return image2



# 图片和语义分割图的主目录
base_data_path = './Zurich_dataset_v1.0'
IMG_path = os.path.join(base_data_path, 'images_tif')
LABEL_path = os.path.join(base_data_path, 'groundtruth')

image = np.zeros([1, 256, 256, 3])
label = np.zeros([1, 256, 256])

cnt_num = 0

for iteration in range(5):
    iteration = iteration + 1
    print("The iteration is %d" % iteration)
    test_img_path = './Zurich_dataset_v1.0/images_tif/zh' + str(iteration) + '.tif'
    test_label_path = './Zurich_dataset_v1.0/groundtruth/zh' + str(iteration) + '_GT.tif'
    OutImageTrainPath = './Lanlingxiang/image3_new_final3.npy'
    OutLabelTrainPath = './Lanlingxiang/annotation3_new_final3.npy'


    # 读取语义分割图，以label的方式也就是(height,width)
    # 是label形式的话将三通道转换成单通道的label形式(像素范围0-8)
    label_test, label_im_proj, label_geotransform = Data_util.ReadImgByPath(test_label_path, is_label=True)
    # 读取图像（wwt:实际遥感图像）
    image_test, image_im_proj, image_geotransform = Data_util.ReadImgByPath(test_img_path)

    # 对图像进行切割做数据增强，size是切割后图片的大小，stride是切割时slide windows的步长
    image_list, label_list, geo_list = Data_util.Crop(image_test, image_geotransform, label_test, size=256, stride=80)

    image_arr_origin = np.array(image_list)[:,1:4,:,:]      # num, channel, width, height
    print("image_arr_origin shape ", image_arr_origin.shape)
    cp_image_arr = image_arr_origin.copy()
    label_arr = np.zeros([1,256,256])
    print("label_arr shape ", label_arr.shape)

    for i in range(0, image_arr_origin.shape[0]):
        # 对每一幅子图进行进行处理
        cnt_num += 1
        if random.randint(0,2):
            patch_arr = PatchGenerator1()   # 256*256
            for x in range(1, image_arr_origin.shape[1]):   # 1,2,3
                for a in range(0, 256):
                    for b in range(0, 256):
                        if patch_arr[a][b] == 1:
                            cp_image_arr[i][x][a][b] = random.randint(0, 65536)
            patch_arr = patch_arr.reshape(1,256,256)
            label_arr = np.concatenate((label_arr, patch_arr), axis=0)
            # print("patch_arr shape ", patch_arr.shape, label_arr.shape)
        else:
            patch_arr = PatchGenerator2()  # 256*256
            for x in range(1, image_arr_origin.shape[1]):  # 1,2,3
                for a in range(0, 256):
                    for b in range(0, 256):
                        if patch_arr[a][b] == 1:
                            cp_image_arr[i][x][a][b] = random.randint(0, 65536)
            patch_arr = patch_arr.reshape(1, 256, 256)
            label_arr = np.concatenate((label_arr, patch_arr), axis=0)
            # print("patch_arr shape ", patch_arr.shape, label_arr.shape)


    print("patch_arr :",cp_image_arr.shape,image_arr_origin.shape)

    image_arr = np.abs(image_arr_origin - cp_image_arr) # 差值的绝对值
    image_arr = np.rollaxis(image_arr, 1, 4)    # x,256,256,3

    print("small batch shape: ", image_arr.shape, label_arr.shape)

        # 结束每个子图的所有分割的循环

    label_arr = label_arr[1:,:,:]
    image = np.append(image,image_arr, axis=0)
    label = np.append(label,label_arr, axis=0)

    print("current image and label shape", image.shape, label.shape)
    # 结束20个循环



# 收入所有train/annotation

print(image.shape,image.dtype)

min, max = np.max(image), np.min(image)

image = (image-min)/(max-min)

print("="*100)
print(cnt_num)
# image = np.array(image)

print(image.shape)
#
# image = image.reshape(cnt_num,256,256,6)
#
#
# label = np.array(label)

print(label.shape)

image_int = image[1:]
label = label[1:]

label.astype('uint8')
image = image.astype('float32')

print("overall image shape ", image.shape, image.dtype, " label shape ", label.shape)



np.save(OutImageTrainPath, image)

np.save(OutLabelTrainPath, label)




    # geo_list = np.array(geo_list)
    #
    # tmp_arr = image_arr.copy()


    # ---将crop后的图像保存---

    # # 图像参数
    # img_num = image_arr.shape[0]
    # im_width = 256
    # im_height = 256
    # im_bands = 6
    # img_datatype = gdal.GDT_UInt16
    #
    #
    # # 标签参数
    # label_num = label_arr.shape[0]
    # label_width = 256
    # label_height = 256
    # label_bands = 1
    # label_datatype = gdal.GDT_Byte

    # # 保存图像
    # for num1 in range(img_num):
    #     driver = gdal.GetDriverByName("GTiff")
    #     dataset = driver.Create(OutImageTrainPath + str(num1) + '.tiff', im_width, im_height, im_bands, img_datatype)
    #     dataset.SetGeoTransform(tuple(geo_list[num1]))  # 写入仿射变换参数
    #     dataset.SetProjection(image_im_proj)  # 写入投影
    #     for i in range(im_bands):
    #         dataset.GetRasterBand(i + 1).WriteArray(image_arr[num1, i, :, :])
    #     del dataset
    #
    # # 保存标签（单通道）
    # for num2 in range(label_num):
    #     driver = gdal.GetDriverByName("GTiff")
    #     dataset = driver.Create(OutLabelTrainPath + str(num2) + '.tiff', label_width, label_height, label_bands,
    #                             label_datatype)
    #     dataset.SetGeoTransform(tuple(geo_list[num2]))  # 写入仿射变换参数
    #     dataset.SetProjection(label_im_proj)  # 写入投影
    #     for i in range(label_bands):
    #         dataset.GetRasterBand(i + 1).WriteArray(label_arr[num2, :, :])
    #     del dataset

print("Finish!")
