import numpy as np
import cv2
import glob
import itertools
import gdal
from sklearn.preprocessing import LabelBinarizer
import os


# ---------------------------------------------------------------------------
# 处理多光谱图像的部分

def getImageArr(path, width, height, imgNorm="sub_mean", odering='channels_first'):
    try:
        dataset = gdal.Open(path)
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        bands = dataset.RasterCount
        img = [dataset.GetRasterBand(i).ReadAsArray(0, 0, cols, rows) for i in range(1, bands + 1)]
        img = np.array(img)
        return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img


def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (width, height))
        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)

    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,
                               output_width, tim):
    # 排除测试集图片
    if tim < 10:
        im_path = images_path + '[!' + str(tim) + ']'
        se_path = segs_path + '[!' + str(tim) + ']'
    elif tim >= 10:
        tim = int(tim % 10)
        im_path = images_path + '1' + '[!' + str(tim) + ']'
        se_path = segs_path + '1' + '[!' + str(tim) + ']'
    elif tim == 20:
        im_path = images_path + '[!2]'
        se_path = segs_path + '[!2]'

    print("The Training image Path:")
    print(im_path + "?*.tiff")
    print(se_path + "?*.tiff")

    images = glob.glob(im_path + "?*.tiff") + glob.glob(im_path + "?*.tif")
    images.sort()
    segmentations = glob.glob(se_path + "?*.tiff") + glob.glob(se_path + "?*.tif")
    segmentations.sort()

    assert len(images) == len(segmentations)
    #    for im, seg in zip(images, segmentations):
    #         assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = zipped.__next__()
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
