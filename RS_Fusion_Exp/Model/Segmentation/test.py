import numpy as np
import os

class BatchDatset:
    imgs = []
    lables = []
    batch_size = 0
    cur_imgs = []
    cur_labels = []
    cur_batch = 0 # index of batch generated
    cur_ind = 0 # index of current image in imgs
    img_width = 256
    img_height = 256

    def __init__(self, imgs_path, label_path, batch_size=2):
        self.imgs = np.load(imgs_path)
        self.lables = np.load(label_path)
        self.batch_size = batch_size
        self.cur_imgs, self.cur_labels = self.imgs[0], self.lables[0].reshape(img_width, img_height, 1)

    def next_batch(self):
        while len(self.cur_imgs) < self.batch_size: # if not enough, get the next image
            self.cur_ind += 1
            #print('appending', self.cur_ind)
            if self.cur_ind >= len(self.imgs):
                #print('leaving', self.cur_ind)
                break
            tmp_imgs = self.imgs[self.cur_ind]
            tmp_labels = self.lables[self.cur_ind]
            self.cur_imgs += tmp_imgs
            self.cur_labels += tmp_labels
        if len(self.cur_imgs) >= self.batch_size:
            #print('getting', self.cur_ind)
            rimat = np.zeros((self.batch_size, self.img_height, self.img_width, 6), dtype=np.float)
            ramat = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.int)
            self.cur_batch += 1 # output a new batch
            for i in range(self.batch_size):
                rimat[i] = self.cur_imgs.pop(0)
                ramat[i, :, :, 0] = self.cur_labels.pop(0)

            return rimat, ramat
        return [], []



if __name__ == '__main__':
    data = BatchDatset('data/trainlist.mat')
    '''ri, ra = data.next_batch()
    while len(ri) != 0:
        ri, ra = data.next_batch()
        print(np.sum(ra))'''
    imgs, labels = data.get_variations(47)
    cnt = 0
    for img in imgs:
        mat = np.zeros(img.shape, dtype=np.int)
        h, w, _ = img.shape
        for i in range(h):
            for j in range(w):
                mat[i][j][0] = round(img[i][j][2] * 255 + 122.675)
                mat[i][j][1] = round(img[i][j][1] * 255 + 116.669)
                mat[i][j][2] = round(img[i][j][0] * 255 + 104.008)
        im = Image.fromarray(np.uint8(mat))
        im.save('img-'+str(cnt)+'.jpg')
        cnt += 1