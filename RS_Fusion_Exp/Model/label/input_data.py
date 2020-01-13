import numpy as np
import Data_util
import time
import matplotlib.pyplot as plt


class Bounding_box:
    def __init__(self, img_array):
        self.img_array = img_array
        self.count = 0  # 接下来是第几个框
        self.label_dict = {}  # 第几个框 count ： 颜色RGB
        self.point_dict = {}  # 点坐标{x},{y}： 属于哪个框 -1 白色
        self.results = []  # 哪个框的所有坐标
        self.drop_index = []
        self.width = img_array.shape[0]
        self.height = img_array.shape[1]
        self.iswhite = np.zeros((self.width, self.height))  # 是否是白色
        self.mask = []
        self.bbox = []
        self.label = []

    def __get_label(self, arr):

        if (arr == [0, 0, 0]).all():
            return 1
        if (arr == [100, 100, 100]).all():
            return 2
        if (arr == [0, 125, 0]).all():
            return 3
        if (arr == [0, 255, 0]).all():
            return 4
        if (arr == [150, 80, 0]).all():
            return 5
        if (arr == [0, 0, 150]).all():
            return 6
        if (arr == [255, 255, 0]).all():
            return 7
        if (arr == [150, 150, 255]).all():
            return 8
        raise ValueError("存在其他类")

    def __mark_white(self, _w, _h):
        self.iswhite[_w, _h] = 1
        # print("{},{} is 白色".format(_w, _h))
        self.point_dict['{}-{}'.format(_w, _h)] = -1

    def __is_white(self, _w, _h):
        if (self.img_array[_w, _h, :] == [255, 255, 255]).all():
            return True
        else:
            return False

    def __create_box(self, _w, _h):
        # 新建框
        self.label_dict[self.count] = self.__get_label(self.img_array[_w, _h, :])
        self.point_dict['{}-{}'.format(_w, _h)] = self.count
        # print('{}-{} is {}  {}'.format(_w, _h, self.count, self.img_array[_w, _h, :]))
        self.results.append([])
        self.results[self.count].append([_w, _h])
        self.count += 1

    def __point_add_box(self, _w, _h, dire):
        # print(point_dict)
        if dire == 'up':
            _count = self.point_dict['{}-{}'.format(_w, _h - 1)]
            self.point_dict['{}-{}'.format(_w, _h)] = _count
            # print('{}-{} is {}  {}'.format(_w, _h, self.count, self.img_array[_w, _h, :]))
            self.results[_count].append([_w, _h])
        elif dire == 'left':
            _count = self.point_dict['{}-{}'.format(_w - 1, _h)]
            self.point_dict['{}-{}'.format(_w, _h)] = _count
            # print('{}-{} is {}  {}'.format(_w, _h, self.count, self.img_array[_w, _h, :]))
            self.results[_count].append([_w, _h])
        elif dire == 'all':
            _count_up = self.point_dict['{}-{}'.format(_w, _h - 1)]
            _count_left = self.point_dict['{}-{}'.format(_w - 1, _h)]
            self.point_dict['{}-{}'.format(_w, _h)] = _count_up
            if _count_up == _count_left:
                # 框也一样
                self.results[_count_up].append([_w, _h])
            else:
                self.results[_count_up] += self.results[_count_left]
                self.results[_count_up].append([_w, _h])
                for point in self.results[_count_left]:
                    self.point_dict['{}-{}'.format(point[0], point[1])] = _count_up
                    # print('{}-{} is {}  {}'.format(point[0], point[1], self.count, self.img_array[_w, _h, :]))
                    self.drop_index.append(_count_left)

    def __same_as(self, dire, _w, _h):
        if dire == 'up':
            if (self.img_array[_w, _h, :] == self.img_array[_w, _h - 1, :]).all():
                return True
            else:
                return False
        if dire == 'left':
            if (self.img_array[_w, _h, :] == self.img_array[_w - 1, _h, :]).all():
                return True
            else:
                return False

    def getBBOXandLabel(self):
        for h in range(self.height):
            for w in range(self.width):
                # print("现在处理{}-{} {}".format(w, h, self.img_array[w, h, :]))
                if self.__is_white(w, h):
                    self.__mark_white(w, h)
                    continue
                if w == 0 and h == 0:
                    self.__create_box(w, h)
                    continue
                if h == 0:
                    if self.__same_as('left', w, h):
                        self.__point_add_box(w, h, 'left')
                    else:
                        self.__create_box(w, h)
                    continue
                if w == 0:
                    if self.__same_as('up', w, h):
                        self.__point_add_box(w, h, 'up')
                    else:
                        self.__create_box(w, h)
                    continue
                if not self.__same_as('up', w, h) and not self.__same_as('left', w, h):
                    # 和上面 左边都不一样
                    # print("都不一样")
                    self.__create_box(w, h)
                elif self.__same_as('up', w, h) and self.__same_as('left', w, h):
                    # print("都一样")
                    # 都一样
                    self.__point_add_box(w, h, 'all')
                elif self.__same_as('left', w, h):
                    # print("和左一样")
                    self.__point_add_box(w, h, 'left')
                elif self.__same_as('up', w, h):
                    # print("和上一样")
                    self.__point_add_box(w, h, 'up')

        for i in range(self.count):
            if i in self.drop_index:
                continue
            minx, miny = np.min(self.results[i], 0)
            maxx, maxy = np.max(self.results[i], 0)
            label = self.label_dict[i]
            self.bbox.append([minx, miny, maxx, maxy])
            self.label.append(label)
            temp_mark = np.zeros((self.width, self.height))
            for point in self.results[i]:
                temp_mark[point[0]][point[1]] = 1
            self.mask.append(temp_mark)
        return np.array(self.mask), np.array(self.bbox), np.array(self.label)


class input_data():
    def __init__(self, IMG_PATH, GT_PATH):
        """
        :param IMG_PATH: 4通道图像路径
        :param GT_PATH:  对应GroundTruth图像路径
        """
        self.GT_PATH = GT_PATH
        self.IMG_PATH = IMG_PATH
        self.masks = []
        self.bboxes = []
        self.labels = []

    def getALL(self, size=256, stride=80):
        """
        :param size: 切割大小
        :param stride: 切割步长
        :return: 返回 原图数组，Groundtruth数组，所有mark，所有bbox （minx,miny,maxx,maxy），所有label
        """
        self.IMG = Data_util.ReadImgByPath(self.IMG_PATH)
        self.GT = Data_util.ReadImgByPath(self.GT_PATH, is_label=False)
        image_list, label_list = Data_util.Crop(self.IMG, self.GT, size=size, stride=stride)
        for i, img in enumerate(label_list):
            start = time.clock()
            img = np.transpose(img, (1, 2, 0))
            mark, bbox, label = Bounding_box(img).getBBOXandLabel()
            self.masks.append(mark)
            self.bboxes.append(bbox)
            self.labels.append(label)
            print("{}/{}  还需 {} s".format(i + 1, len(label_list), (time.clock() - start) * (len(label_list) - i - 1)))
        return image_list, label_list, self.masks, self.bboxes, self.labels


if __name__ == '__main__':
    for i in range(20):
        print("第{}个".format(i + 1))
        data = input_data('Zurich_dataset_v1.0/images_tif/zh{}.tif'.format(i + 1),
                          'Zurich_dataset_v1.0/groundtruth/zh{}_GT.tif'.format(i + 1))
        imgs, labs, masks, bboxes, labels = data.getALL()
        # print(masks[0])
        print("保存", i + 1)
        np.savez_compressed('result/image_{}.npz'.format(i + 1), images=imgs, masks=masks, bboxes=bboxes,
                            labels=labels)
        print("保存完成", i + 1)
