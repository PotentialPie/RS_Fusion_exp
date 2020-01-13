import numpy as np
from PIL import Image

img = Image.open('zh1_GT.jpg')
img_array = np.array(img)[:100, :100, :]
width = img_array.shape[0]
height = img_array.shape[1]
print(img_array.shape)

visited = np.zeros((width, height))


class bounding_box:
    def __init__(self, minx, miny, maxx, maxy, color):
        self.min_x = minx
        self.min_y = miny
        self.max_x = maxx
        self.max_y = maxy
        self.color = color

    def update_point(self, x, y):
        if x < self.min_x:
            self.min_x = x
        if x > self.max_x:
            self.max_x = x
        if y < self.min_y:
            self.min_y = y
        if y > self.max_y:
            self.max_y = y

    def update_box(self, box):
        # 默认一定相切或相交
        if box.min_x < self.min_x:
            self.min_x = box.min_x
        if box.max_x > self.max_x:
            self.max_x = box.max_x
        if box.min_y < self.min_y:
            self.min_y = box.min_y
        if box.max_y > self.max_y:
            self.max_y = box.max_y


def find_bound(w, h, box):
    # 如果超过边界 不更新
    if w < 0 or w >= width or h < 0 or h >= height:
        return box
    # 如果访问过 不更新
    # print(w,width,h,height)
    if visited[w, h] == 1:
        return box
    # 如果是白色，标记访问，不更新
    if box.color.all() == np.array([255, 255, 255]).all():
        visited[w, h] = 1
        return box
    # 如果不是相同颜色 不更新
    if img_array[w, h, :].all() != box.color.all():
        return box
    # 如果是相同颜色
    visited[w, h] = 1
    box.update_point(w, h)
    return merge_box([find_bound(w + 1, h, box),
                      find_bound(w - 1, h, box),
                      find_bound(w, h + 1, box)
                      ], box.color)


def merge_box(boxs, color):
    result_box = bounding_box(width, height, 0, 0, color)
    for box in boxs:
        result_box.update_box(box)
    return result_box


if __name__ == '__main__':
    result = []
    for w in range(width):
        for h in range(height):
            res = find_bound(w, h, bounding_box(w, h, w, h, img_array[w, h, :]))
            if res.min_x == w and res.max_x == w and res.min_y == h and res.max_y == h:
                continue
            else:
                result.append(res)
    print(result)
