import cv2 as cv
import numpy as np
from time import time
from PIL import Image
from random import choice


def get_valid(img, comment):
    img = img[:, :, 0]
    img = img / 255
    comment = comment[:, :, 0]

    i_h, i_w = img.shape
    c_h, c_w = comment.shape

    valid_set = set()
    salient_set = set()

    for x in range(i_h):
        for y in range(i_w):
            # 二值化
            img[x, y] = int(round(img[x, y]))

            # 有效点: 0点 和 不越界的点
            if img[x, y] == 0:
                if x + c_h >= i_h or y + c_w >= i_w:
                    continue
                valid_set.add((x, y))

            # 无效点: 1点
            if img[x, y] == 1:
                salient_set.add((x, y))


    temp_valid_set = set()

    print(len(valid_set))
    count = 0
    for x0, y0 in valid_set:
        comment_set = set()
        for x in range(c_h):
            for y in range(c_w):
                comment_set.add((x0 + x, y0 + y))
        if len(comment_set & salient_set) == 0:
            temp_valid_set.add((x0, y0))
        count += 1

        if count % 10000 == 0:
            print(count)


    valid_set = temp_valid_set
    img = np.full((i_h, i_w), 255)

    for x, y in valid_set:
        img[x, y] = 0

    return np.expand_dims(img, axis=2).repeat(3, axis=2)

def do_get_valid():
    start_time = time()

    img_path = '/home/xuyifang/VST/preds/XYF/010.png'
    img = cv.imread(img_path)
    comment = np.ones([20, 300, 3])

    img = get_all_valid(img, comment)

    save_path = '/home/xuyifang/VST/preds/XYF/010_.png'
    cv.imwrite(save_path, img)

    print('time: {:.2f}s'.format((time() - start_time)))


def do_random_insert(img, comment):
    """get (x, y) of top-left pixel
    """
    img = img[:, :, 0]
    img = img / 255
    comment = comment[:, :, 0]

    i_h, i_w = img.shape
    c_h, c_w = comment.shape

    valid_set = set()
    salient_set = set()

    for x in range(i_h):
        for y in range(i_w):
            # 二值化
            img[x, y] = int(round(img[x, y]))

            # 有效点: 0点 和 不越界的点
            if img[x, y] == 0:
                if x + c_h >= i_h or y + c_w >= i_w:
                    continue
                valid_set.add((x, y))

            # 无效点: 1点
            if img[x, y] == 1:
                salient_set.add((x, y))

    while valid_set:
        x0, y0 = choice(list(valid_set))
        valid_set.remove((x0, y0))
        
        temp_set = set()
        if c_w > 10 and c_h > 10:
            for x in range(10, c_h, 10):
                for y in range(10, c_w, 10):
                    temp_set.add((x0 + x, y0 + y))
        temp_set.add((x0, y0 + c_w))
        temp_set.add((x0 + c_h, y0))
        temp_set.add((x0 + c_h, y0 + c_w))
        temp_set.add((x0 + c_h // 2, y0 + c_w // 2))

        for x, y in temp_set:
            if (x, y) in salient_set:
                continue
        
        comment_set = set()

        for x in range(c_h):
            for y in range(c_w):
                comment_set.add((x0 + x, y0 + y))
        if len(comment_set & salient_set) == 0:
            return x0, y0

        if len(valid_set) % 10000 == 0:
            print(len(valid_set))

    return -1, -1

def random_insert(img_path, comment_path, save_path):
    start_time = time()
    print('load image: {}'.format(img_path))

    img = cv.imread(img_path)
    comment = cv.imread(comment_path)
    
    # 随机获得左上角的位置
    x0, y0 = do_random_insert(img, comment)  
    
    if x0 == y0 == -1:
        cv.imwrite(save_path, img)
    else:
        for x in range(comment.shape[0]):
            for y in range(comment.shape[1]):
                img[x0 + x, y0 + y] = comment[x, y]
        cv.imwrite(save_path, img)
    
    print('speed time: {:.2f}s'.format((time() - start_time)))

def insert_name_card():
    pass

def insert_video_title(img_path, vt_path, save_path):
    img = cv.imread(img_path)
    vt = cv.imread(vt_path)
    
    # 直接插入到左上角
    for x in range(int(vt.shape[0])):
        for y in range(int(vt.shape[1])):
            img[x, y] = vt[x, y]
    cv.imwrite(save_path, img)


def video_random_insert():
    pass

