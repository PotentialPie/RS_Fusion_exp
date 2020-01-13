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


def find_bound(w, h, pre_color, points):
    # 如果超过边界 不更新
    if w < 0 or w >= width or h < 0 or h >= height:
        print("超过边界")
        return points
    # 如果访问过 不更新
    # print(w,width,h,height)
    if visited[w, h] == 1:
        print("访问过")
        return points
    # 如果是白色，标记访问，不更新
    if img_array[w, h, :][0] == 255 and img_array[w, h, :][1] == 255 and img_array[w, h, :][2] == 255:
        print("是白色", img_array[w, h, :], np.array([255, 255, 255]))
        visited[w, h] = 1
        return points
    # 如果不是相同颜色 不更新
    if img_array[w, h, :].all() != pre_color.all():
        print("不同颜色", img_array[w, h, :], pre_color)
        return points
    # 如果是相同颜色
    visited[w, h] = 1
    points.append([w, h])
    print("原本points")
    print(points)
    print("向右")
    print(find_bound(w + 1, h, pre_color, points))
    print("向左")
    print(find_bound(w - 1, h, pre_color, points))
    print("向下")
    print(find_bound(w, h + 1, pre_color, points))
    return points + find_bound(w + 1, h, pre_color, points) + find_bound(w - 1, h, pre_color, points) + find_bound(w, h + 1, pre_color, points)


def merge_box(boxs, color):
    result_box = bounding_box(width, height, 0, 0, color)
    for box in boxs:
        result_box.update_box(box)
    return result_box


if __name__ == '__main__':
    result = []
    for w in range(width):
        for h in range(height):
            res = find_bound(w, h, img_array[w, h, :], [])
            print(res)
            if len(res) == 0:
                continue
            else:
                minx, miny = np.min(res, axis=0)
                maxx, maxy = np.max(res, axis=0)
                box = bounding_box(minx, miny, maxx, maxy, img_array[w, h, :])
                result.append(box)
    print(result)
