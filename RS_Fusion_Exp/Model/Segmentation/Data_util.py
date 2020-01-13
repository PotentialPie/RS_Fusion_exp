import numpy as np
import random
import os

import gdal

from sklearn.preprocessing import LabelBinarizer

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


def ReadFileList(File_dir):
    """
    Read file list
    :param File_dir: the base dir
    :return: list of filenames
    """
    return os.listdir(File_dir)


def ReadImgByPath(IMG_path, is_label=False):
    """
    读取单个文件的方法，由于Zurich的数据格式是geotiff，不能用cv2或者PIL库来读取
    需要用专门的python-gdal库来读
    :param IMG_path: 图片地址
    :param is_label: 是否是label图像
    :return: 图片的numpy形式（未做处理）
    """
    # 这里可能会出现类似于invalid TIFF directory; tags are not sorted in ascending order之类的警告，不用管
    dataset = gdal.Open(IMG_path)
    # 接着可以从dataset中获取一些基础属性
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    # 文件类型
    driver = dataset.GetDriver().LongName
    # print(cols, rows, bands, driver)
    # 利用GetGeoTransform获取地图信息, 分别是左上X轴坐标，东西向分辨率，旋转角度，左上Y轴坐标，旋转角度，南北向分辨率
    geotransform = dataset.GetGeoTransform()
    # 地图投影信息
    im_proj = dataset.GetProjection()
    # print(geotransform)
    # 根据bands来还原图片
    IMG = [dataset.GetRasterBand(i).ReadAsArray(0, 0, cols, rows) for i in range(1, bands + 1)]
    IMG = np.array(IMG)
    # print('shape:' + str(IMG.shape))
    # 如果是label，就做转换
    if is_label:
        Label = np.zeros((IMG.shape[1], IMG.shape[2]), dtype=int)
        # print(Label.shape)
        for i in range(len(LEGEND)):
            for m in range(IMG.shape[1]):
                for n in range(IMG.shape[2]):
                    if IMG[0][m][n] == LEGEND[i][0] and IMG[1][m][n] == LEGEND[i][1] and IMG[2][m][n] == LEGEND[i][2]:
                        Label[m][n] = i
        return Label, im_proj, geotransform
    return IMG, im_proj, geotransform


def ReadImgList(File_dir, is_label=False):
    """
    根据图片的根目录，读取当前目录下的所有图片，并放入list
    :param File_dir: 图片所在目录
    :return: list格式的图片列表，每个item是一个numpy对象
    """
    # 读取图片的地址列表
    File_list = ReadFileList(File_dir)
    # print(len(File_list))
    # 根据图片名称中的数字进行排序
    File_list = sorted(File_list, key=lambda d: int(d.split('.')[0].split('_')[0][2:4]))
    # print(File_list[:10])
    # 用于保存图片列表
    IMG_list = []

    # 遍历图片地址列表，读取图片append到IMG_list中
    for IMG_file in File_list:
        IMG_id = str(IMG_file).split('.')[0]
        # print(IMG_id)
        IMG_file = os.path.join(File_dir, IMG_file)
        IMG = ReadImgByPath(IMG_file, is_label)
        print(IMG.shape)
        IMG_list.append(IMG)
    return IMG_list


def Count(seg_label):
    """
    根据给定的语义分割图，计算label的个数分布
    :param seg_label: 给定语义分割图
    :return: dict对象
    """
    print("Counting, please wait......")
    # 如果是label对象，则进行计算
    if len(seg_label.shape) == 2:
        minlabel = 0
        maxlabel = 8
        label_dict = {}
        for label in range(minlabel, maxlabel + 1):
            num = np.sum(seg_label == label)
            print("The count of label " + str(label) + " is " + str(num))
            label_dict[str(label)] = num
        print("Counted successfully!")
        return label_dict
    # 不是label对象，则报错
    else:
        print("This is image, not label!")


def one_hot_encode(x, numOfLabel):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(numOfLabel))
    width = x.shape[0]
    height = x.shape[1]
    x = x.reshape(-1, )
    x = label_binarizer.transform(x)
    x = x.reshape(width, height, numOfLabel)
    return x


def Crop(image, image_geotransform, seg_label, size=224, stride=20):
    """
    根据图像和语义分割图，进行crop
    :param image: 给定语义分割图
    :param seg_label: 给定语义分割图
    :param size: 切割的图片大小
    :param stride: 切割步长
    :return: image_list,label_list
    """
    count = 0
    image_list = []
    label_list = []
    geo_list = []
    start_row = 0
    start_col = 0

    Xsize = seg_label.shape[1]
    Ysize = seg_label.shape[0]

    # 分辨率
    XRes = image_geotransform[1]  # width
    YRes = image_geotransform[5]  # height

    # 左上角
    LUX = image_geotransform[0]
    LUY = image_geotransform[3]
    # 右上角
    RUX = image_geotransform[0] + Xsize * XRes
    RUY = image_geotransform[3]
    # 左下角
    LBX = image_geotransform[0]
    LBY = image_geotransform[3] + Ysize * YRes
    # 右下角
    RBX = image_geotransform[0] + Xsize * XRes
    RBY = image_geotransform[3] + Ysize * YRes

    # 输出地理参数
    o_geotransform = np.zeros(6)
    o_geotransform[1] = XRes
    o_geotransform[5] = YRes

    tempX = LUX
    tempY = LUY
    while start_row + size < seg_label.shape[0]:
        end_row = start_row + size
        o_geotransform[3] = tempY
        while start_col + size < seg_label.shape[1]:
            end_col = start_col + size
            print(str(count) + ' ' + str(start_row) + ' ' + str(end_row) + ' ' + str(start_col) + ' ' + str(end_col))
            img_feature = image[:, start_row:end_row, start_col:end_col]
            img_label = seg_label[start_row:end_row, start_col:end_col]
            o_geotransform[0] = tempX
            image_list.append(img_feature)
            label_list.append(img_label)
            geo_list.append(np.array(o_geotransform))
            # 更新
            start_col = start_col + stride
            tempX = tempX + stride * XRes
            count = count + 1
            # 判断是否到了X轴边界
            if start_col + size > seg_label.shape[1]:  # 右边残余图像
                end_col = seg_label.shape[1]
                start_col = end_col - size
                print(str(count) + ' ' + str(start_row) + ' ' + str(end_row) + ' ' + str(start_col) + ' ' + str(end_col))
                img_feature = image[:, start_row:end_row, start_col:end_col]
                img_label = seg_label[start_row:end_row, start_col:end_col]
                o_geotransform[0] = RUX - size * XRes
                image_list.append(img_feature)
                label_list.append(img_label)
                geo_list.append(np.array(o_geotransform))
                count = count + 1
        start_row = start_row + stride
        tempY = tempY + stride * YRes
        start_col = 0
        tempX = LUX

    # 计算Y轴多出来部分
    end_row = seg_label.shape[0]
    start_row = end_row - size
    o_geotransform[3] = LBY - YRes * size
    while start_col + size < seg_label.shape[1]:
        end_col = start_col + size
        print(str(count) + ' ' + str(start_row) + ' ' + str(end_row) + ' ' + str(start_col) + ' ' + str(end_col))
        img_feature = image[:, start_row:end_row, start_col:end_col]
        img_label = seg_label[start_row:end_row, start_col:end_col]
        o_geotransform[0] = tempX
        image_list.append(img_feature)
        label_list.append(img_label)
        geo_list.append(np.array(o_geotransform))
        # 更新
        start_col = start_col + stride
        tempX = tempX + stride * XRes
        count = count + 1
        # 判断是否到了X轴边界(右下角)
        if start_col + size > seg_label.shape[1]:
            end_col = seg_label.shape[1]
            start_col = end_col - size
            print(str(count) + ' ' + str(start_row) + ' ' + str(end_row) + ' ' + str(start_col) + ' ' + str(end_col))
            img_feature = image[:, start_row:end_row, start_col:end_col]
            img_label = seg_label[start_row:end_row, start_col:end_col]
            o_geotransform[0] = RUX - size * XRes
            image_list.append(img_feature)
            label_list.append(img_label)
            geo_list.append(np.array(o_geotransform))
            count = count + 1

    c = list(zip(image_list, label_list, geo_list))
    #    random.shuffle(c)
    image_list[:], label_list[:], geo_list[:] = zip(*c)
    return image_list, label_list, geo_list
