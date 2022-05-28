import os
import shutil
import sys
import time
from random import choice, randint

import cv2 as cv
import imageio
import numpy as np


def count_time(func):
    def _count_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("'{}' speed time: {:.1f}s".format(func.__name__, end - start))

    return _count_time


@count_time
def input_video_to_input_img(input_video_path, output_image_path):
    vc = cv.VideoCapture(input_video_path)
    fps = vc.get(cv.CAP_PROP_FPS)
    print('video fps: {}'.format(fps))

    vc_isOpened = vc.isOpened()
    video_time = 0

    while 1:
        vc_isOpened, image = vc.read()

        if vc_isOpened:
            save_path = os.path.join(output_image_path, str(video_time).rjust(5, '0') + '.png')
            cv.imwrite(save_path, image)
            video_time += 1
            cv.waitKey(1)
        else:
            break

    vc.release()
    print('input images: {}'.format(video_time))


@count_time
def input_img_to_sod_input_img(root_path, input_image_path):
    # /home/v-yifangxu/Desktop/VST
    parent_root_path = os.path.abspath(root_path + os.path.sep + "..")
    # /home/v-yifangxu/Desktop/VST/train_test_eval.py
    python_file_path = os.path.join(parent_root_path, 'train_test_eval.py')

    # /home/v-yifangxu/Desktop/VST/data/res/FPJpUE407-w
    parent_input_image_path = os.path.abspath(input_image_path + os.path.sep + "..")
    # FPJpUE407-w
    dataset_name = parent_input_image_path.split('/')[-1]
    # /home/v-yifangxu/Desktop/VST/data/res
    parent_input_image_path = os.path.abspath(parent_input_image_path + os.path.sep + "..")

    os.system("python " + python_file_path + \
              " --data_root " + parent_input_image_path + "/" +
              " --save_test_path_root " + parent_input_image_path + "/" +
              " --test_paths " + dataset_name + \
              " --Testing True")


def get_sod_set(sod, sod_ratio=0.2):
    sod = sod[:, :, 0]
    sod = sod / 255

    index = np.array(np.where(sod > sod_ratio)).transpose()
    sod_set = set(tuple(x) for x in index.tolist())

    x_min = np.min(index[:, 0]).item()
    x_max = np.max(index[:, 0]).item()
    y_min = np.min(index[:, 1]).item()
    y_max = np.max(index[:, 1]).item()

    return sod_set, x_min, x_max, y_min, y_max


def get_iou(a, b):
    a_set, *_ = get_sod_set(a)
    b_set, *_ = get_sod_set(b)

    overlap = a_set & b_set
    union = a_set | b_set

    return len(overlap) / len(union)


def insert(img, mask, x0, y0, mode=2):
    def insert_(img, mask, x0, y0):
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                img[x0 + x, y0 + y] = mask[x, y]
        return img

    i_h, i_w, _ = img.shape
    m_h, m_w, _ = mask.shape

    # 右下
    if mode == 0:
        if x0 - m_h >= 0 and y0 - m_w >= 0:
            return insert_(img, mask, x0 - m_h, y0 - m_w)

    # 左下
    if mode == 1:
        if x0 - m_h >= 0 and y0 + m_w < i_w:
            return insert_(img, mask, x0 - m_h, y0)

    # 左上
    if mode == 2:
        if x0 + m_h < i_h and y0 + m_w < i_w:
            return insert_(img, mask, x0, y0)

    # 右上
    if mode == 3:
        if x0 + m_h < i_h and y0 - m_w >= 0:
            return insert_(img, mask, x0, y0 - m_w)

    return None


def insert_video_title(img, sod, vt):
    img = insert(img, vt, 0, 0)
    sod = insert(sod, vt, 0, 0)
    return img, sod


def insert_comment(img, sod, comment, vt):
    vt_w = vt.shape[1]

    img = insert(img, comment, 0, int(vt_w * 1.2))
    sod = insert(sod, comment, 0, int(vt_w * 1.2))

    return img, sod


def insert_name_card(img, sod, sod_i, nc, scale=0.25):
    sod_set, x_min, x_max, y_min, y_max = get_sod_set(sod_i)

    ratio = (y_max - y_min) / (x_max - x_min)

    if ratio <= 1:
        x_add = 1 / ratio
        y_add = 1
    else:
        x_add = 1
        y_add = ratio

    # 从中心遍历到右上 ↗    找出不在sod上的第2个点
    x = x_min + (x_max - x_min) / 2
    y = y_min + (y_max - y_min) / 2
    count = 0

    while x >= x_min and y <= y_max:
        x -= x_add
        y += y_add

        x_int, y_int = int(x), int(y)

        if (x_int, y_int) not in sod_set:
            count += 1
            if count >= 2:
                break

    x, y = int(x), int(y)

    # 插入的位置
    x -= int(abs(x - x_min) * scale)
    y += int(abs(y_max - y) * scale)

    _img = insert(img, nc, x, y, mode=1)
    _sod = insert(sod, nc, x, y, mode=1)

    # 如果不能插入在右上角，那就插入在左上角
    if _img is None or _sod is None:
        # 从中心遍历到左上 ↖    找出不在sod上的第2个点
        x = x_min + (x_max - x_min) / 2
        y = y_min + (y_max - y_min) / 2
        count = 0

        while x >= x_min and y >= y_min:
            x -= x_add
            y -= y_add

            x_int, y_int = int(x), int(y)

            if (x_int, y_int) not in sod_set:
                count += 1
                if count >= 2:
                    break

            x, y = int(x), int(y)

            # 插入的位置
            x -= int(abs(x - x_min) * scale)
            y -= int(abs(y - y_min) * scale)

            _img = insert(img, nc, x, y, mode=0)
            _sod = insert(sod, nc, x, y, mode=0)

    if _img is None or _sod is None:
        return img, sod
    else:
        return _img, _sod


@count_time
def insert_all(
        input_image_root,
        output_image_root,
        sod_input_image_root,
        sod_output_image_root,
        vt_path,
        nc_path,
        comment_path,
        iou_ratio=0.6,
):
    vt = cv.imread(vt_path)
    nc = cv.imread(nc_path)
    comment = cv.imread(comment_path)

    image_names = os.listdir(input_image_root)
    image_names.sort(key=lambda x: x)

    image_list = []
    sod_list = []

    n = len(image_names)
    i = 0

    while i < n - 1:
        sod_i = cv.imread(os.path.join(sod_input_image_root, image_names[i]))
        j = i + 1

        while j < n:
            sod_j = cv.imread(os.path.join(sod_input_image_root, image_names[j]))
            iou = get_iou(sod_i, sod_j)

            if iou <= iou_ratio:
                print(i, j, iou)
                break
            j += 1

        for k in range(i, j):
            img_k = cv.imread(os.path.join(input_image_root, image_names[k]))
            sod_k = cv.imread(os.path.join(sod_input_image_root, image_names[i]))

            img_k, sod_k = insert_name_card(img_k, sod_k, sod_i, nc)
            img_k, sod_k = insert_video_title(img_k, sod_k, vt)
            img_k, sod_k = insert_comment(img_k, sod_k, comment, vt)

            image_list.append(img_k)
            sod_list.append(sod_k)

        i = j

    for k in range(n):
        cv.imwrite(os.path.join(output_image_root, image_names[k]), image_list[k])
        cv.imwrite(os.path.join(sod_output_image_root, image_names[k]), sod_list[k])


@count_time
def _insert_all(input_image_root, output_image_root, sod_input_image_root, sod_output_image_root, vt_path, nc_path, comment_path):
    image_names = os.listdir(input_image_root)
    image_names.sort(key=lambda x: x)

    for image_name in image_names:
        input_image_path = os.path.join(input_image_root, image_name)
        output_image_path = os.path.join(output_image_root, image_name)
        sod_input_image_path = os.path.join(sod_input_image_root, image_name)
        sod_output_image_path = os.path.join(sod_output_image_root, image_name)

        img = cv.imread(input_image_path)
        sod = cv.imread(sod_input_image_path)
        vt = cv.imread(vt_path)
        nc = cv.imread(nc_path)
        comment = cv.imread(comment_path)

        img, sod = insert_name_card(img, sod, nc)

        img, sod = insert_video_title(img, sod, vt)

        img, sod = insert_comment(img, sod, comment, vt)

        cv.imwrite(output_image_path, img)
        cv.imwrite(sod_output_image_path, sod)


@count_time
def image_to_video(input_video_path, output_image_root, output_video_root):
    vc = cv.VideoCapture(input_video_path)
    fps = vc.get(cv.CAP_PROP_FPS)

    images = []

    items = os.listdir(output_image_root)
    items.sort(key=lambda x: int(x.split('.')[0]))

    for item in items:
        item = os.path.join(output_image_root, item)
        image = cv.imread(item)
        image = image[..., ::-1]
        images.append(image)

    video = imageio.mimsave(output_video_root, images, 'GIF', duration=1 / fps)


@count_time
def main():
    root = '/home/v-yifangxu/Desktop/VST/data'

    input_video_root = os.path.join(root, 'input')
    if not os.path.exists(input_video_root):
        raise Exception('Please select right input path!')

    output_video_root = os.path.join(root, 'output')
    os.makedirs(output_video_root, exist_ok=True)

    vt_path = os.path.join(root, 'video_title.png')
    nc_path = os.path.join(root, 'name_card.png')
    comment_path = os.path.join(root, 'comment.png')

    video_names = os.listdir(input_video_root)
    video_names.sort(key=lambda x: x)

    for video_name in video_names:
        print("===========[ start prcessing video: {} ]===========".format(video_name))

        res_path = os.path.join(root, 'res', video_name.split('.')[0])
        # /home/v-yifangxu/Desktop/VST/data/res/FPJpUE407-w/
        sod_output_video_root = res_path

        input_image_root = os.path.join(res_path, 'input_image')
        output_image_root = os.path.join(res_path, 'output_image')
        sod_input_image_root = os.path.join(res_path, 'sod_image')
        sod_output_image_root = os.path.join(res_path, 'sod_output_image')

        os.makedirs(input_image_root, exist_ok=True)
        os.makedirs(output_image_root, exist_ok=True)
        os.makedirs(sod_input_image_root, exist_ok=True)
        os.makedirs(sod_output_image_root, exist_ok=True)

        '''input_video_to_input_img'''
        input_video_path = os.path.join(input_video_root, video_name)
        input_video_to_input_img(input_video_path, input_image_root)

        '''input_img_to_sod_img'''
        input_img_to_sod_input_img(root, input_image_root)

        '''insert [video title, name card, comment]'''
        insert_all(input_image_root, output_image_root,
                   sod_input_image_root, sod_output_image_root,
                   vt_path, nc_path, comment_path)

        '''output_img_to_output_video'''
        output_video_path = os.path.join(output_video_root, video_name)
        image_to_video(input_video_path, output_image_root, output_video_path)

        '''output_sod_img_to_output_sod_video'''
        sod_output_video_path = os.path.join(sod_output_video_root, video_name)
        image_to_video(input_video_path, sod_output_image_root, sod_output_video_path)

        print('===========[ finish! ]===========')


if __name__ == "__main__":
    main()
