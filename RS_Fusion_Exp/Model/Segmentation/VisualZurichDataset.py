import glob
import numpy as np
import cv2
import random
import argparse
import matplotlib.pyplot as plt
import gdal


def imageSegmentationGenerator(images_path, segs_path, n_classes):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg") \
             + glob.glob(images_path + "*.tiff") + glob.glob(images_path + "*.tif")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg") \
                    + glob.glob(segs_path + "*.tiff") + glob.glob(segs_path + "*.tif")

    segmentations.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(n_classes)]

    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):
  #      assert (im_fn.split('/')[-1] == seg_fn.split('/')[-1])
        # 读入真实图片
        dataset = gdal.Open(im_fn)
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        bands = dataset.RasterCount
        img = [dataset.GetRasterBand(i).ReadAsArray(0, 0, cols, rows) for i in range(1, bands + 1)]
        img = np.array(img)
        img1 = np.rollaxis(img[0:3, :, :], 0, 3)

        # img = cv2.imread(im_fn)
        # 读入label图片
        seg = cv2.imread(seg_fn)

        '''
        # 当且仅当调试时候使用
        plt.imshow(img1)
        plt.show()
      
        plt.imshow(seg)
        plt.show()
        print("1")
        '''
        '''
        cv2.imshow("img", img1)
        cv2.imshow("seg_img", seg_img)
        cv2.waitKey()
    '''



parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str)
parser.add_argument("--annotations", type=str)
parser.add_argument("--n_classes", type=int)
args = parser.parse_args()

imageSegmentationGenerator(args.images, args.annotations, args.n_classes)
