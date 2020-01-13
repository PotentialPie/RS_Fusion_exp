import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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


# img = Image.open('zh1_GT.jpg')
def get_label(arr):
    # 读出来RGB顺序相反？？？
    if (arr == [0, 0, 0]).all():
        return 1
    if (arr == [100, 100, 100]).all():
        return 2
    if (arr == [0, 125, 0]).all():
        return 3
    if (arr == [0, 255, 0]).all():
        return 4
    if (arr == [0, 80, 150]).all():
        return 5
    if (arr == [150, 0, 0]).all():
        return 6
    if (arr == [0, 255, 255]).all():
        return 7
    if (arr == [255, 150, 150]).all():
        return 8
    raise ValueError("存在其他类")


def mark_white(_w, _h):
    iswhite[_w, _h] = 1
    # print("{},{} is 白色".format(_w, _h))
    point_dict['{}-{}'.format(_w, _h)] = -1


def is_white(_w, _h):
    if (img_array[_w, _h, :] == [255, 255, 255]).all():
        return True
    else:
        return False


def create_box(_w, _h):
    global count
    # 新建框
    label_dict[count] = get_label(img_array[_w, _h, :])
    point_dict['{}-{}'.format(_w, _h)] = count
    # print('{}-{} is {}  {}'.format(_w, _h, count, img_array[_w, _h, :]))
    results.append([])
    results[count].append([_w, _h])
    count += 1


def point_add_box(_w, _h, dire):
    # print(point_dict)
    if dire == 'up':
        _count = point_dict['{}-{}'.format(_w, _h - 1)]
        point_dict['{}-{}'.format(_w, _h)] = _count
        # print('{}-{} is {}  {}'.format(_w, _h, count, img_array[_w, _h, :]))
        results[_count].append([_w, _h])
    elif dire == 'left':
        _count = point_dict['{}-{}'.format(_w - 1, _h)]
        point_dict['{}-{}'.format(_w, _h)] = _count
        # print('{}-{} is {}  {}'.format(_w, _h, count, img_array[_w, _h, :]))
        results[_count].append([_w, _h])
    elif dire == 'all':
        _count_up = point_dict['{}-{}'.format(_w, _h - 1)]
        _count_left = point_dict['{}-{}'.format(_w - 1, _h)]
        point_dict['{}-{}'.format(_w, _h)] = _count_up
        if _count_up == _count_left:
            # 框也一样
            results[_count_up].append([_w, _h])
        else:
            results[_count_up] += results[_count_left]
            results[_count_up].append([_w, _h])
            for point in results[_count_left]:
                point_dict['{}-{}'.format(point[0], point[1])] = _count_up
                # print('{}-{} is {}  {}'.format(point[0], point[1], count, img_array[_w, _h, :]))
            drop_index.append(_count_left)


def same_as(dire, _w, _h):
    if dire == 'up':
        if (img_array[_w, _h, :] == img_array[_w, _h - 1, :]).all():
            return True
        else:
            return False
    if dire == 'left':
        if (img_array[_w, _h, :] == img_array[_w - 1, _h, :]).all():
            return True
        else:
            return False


list = os.listdir('Zurich_dataset_v1.0/groundtruth')
for i,filename in enumerate(list):
    file = filename[:-4]
    filename = 'Zurich_dataset_v1.0/groundtruth/' + filename
    img = cv2.imread(filename, -1)
    img_array = np.array(img)  # [:300, :300, :]

    # print(np.array(img_array))
    width = img_array.shape[0]
    height = img_array.shape[1]
    # print(img_array.shape)

    count = 0  # 接下来是第几个框
    label_dict = {}  # 第几个框 count ： 颜色RGB
    point_dict = {}  # 点坐标{x},{y}： 属于哪个框 -1 白色
    results = []  # 哪个框的所有坐标
    drop_index = []
    iswhite = np.zeros((width, height))  # 是否是白色

    for h in range(height):
        for w in range(width):
            print("{}/{} {}/{}".format(i+1,len(list),h * width + w, height * width))
            if is_white(w, h):
                mark_white(w, h)
                continue
            if w == 0 and h == 0:
                create_box(w, h)
                continue
            if h == 0:
                if same_as('left', w, h):
                    point_add_box(w, h, 'left')
                else:
                    create_box(w, h)
                continue
            if w == 0:
                if same_as('up', w, h):
                    point_add_box(w, h, 'up')
                else:
                    create_box(w, h)
                continue
            if not same_as('up', w, h) and not same_as('left', w, h):
                # 和上面 左边都不一样
                # print("都不一样")
                create_box(w, h)
            elif same_as('up', w, h) and same_as('left', w, h):
                # print("都一样")
                # 都一样
                point_add_box(w, h, 'all')
            elif same_as('left', w, h):
                # print("和左一样")
                point_add_box(w, h, 'left')
            elif same_as('up', w, h):
                # print("和上一样")
                point_add_box(w, h, 'up')

    fig = plt.figure(figsize=(height / 100, width / 100))
    ax = fig.add_subplot(111)
    ax.imshow(img_array)

    for i in range(count):
        if i in drop_index:
            continue
        minx, miny = np.min(results[i], 0)
        maxx, maxy = np.max(results[i], 0)
        label = label_dict[i]
        # x y 相反
        rect = plt.Rectangle((miny, minx), (maxy - miny), (maxx - minx), fill=False, edgecolor='r')
        ax.add_patch(rect)
        # with open('../data/{}_bounding_box.txt'.format(file), 'a', encoding='utf-8') as f:
        #    f.write('{} {} {} {} {}\n'.format(minx, miny, maxx, maxy, label))
    ax.set_xticks([])
    ax.set_yticks([])
    print("保存")
    plt.savefig("result_{}.jpg".format(file))
    plt.close('all')
