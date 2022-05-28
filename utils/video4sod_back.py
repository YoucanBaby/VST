import os
import shutil
import sys
import time
from random import choice, randint

import cv2 as cv
import imageio


def count_time(func):
    def _count_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("'{}' speed time: {:.1f}s".format(func.__name__, end - start))

    return _count_time


@count_time
def video2img(input_video_path, output_image_path):
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
def img2sod(root_path, input_image_path):
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


def img2gif(image_path, video_path, recover_path):
    vc = cv.VideoCapture(video_path)
    fps = vc.get(cv.CAP_PROP_FPS)
    print('fps: {}'.format(fps))

    images = []

    items = os.listdir(image_path)
    items.sort(key=lambda x: int(x.split('.')[0]))

    for item in items:
        item = os.path.join(image_path, item)
        image = cv.imread(item)
        image = image[..., ::-1]
        images.append(image)

    print('total image is {}.'.format(len(images)))
    gif = imageio.mimsave(recover_path, images, 'GIF', duration=1 / fps)

    print('finish!')


def get_salient_set(sod):
    sod = sod[:, :, 0]
    sod = sod / 255

    s_h, s_w = sod.shape
    salient_set = set()
    x_min, x_max = sys.maxsize, 0
    y_min, y_max = sys.maxsize, 0

    for x in range(s_h):
        for y in range(s_w):
            # 二值化
            sod[x, y] = int(round(sod[x, y]))

            # 无效点: 1点
            if sod[x, y] == 1:
                salient_set.add((x, y))
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

    return salient_set, x_min, x_max, y_min, y_max


def insert(img, mask, x0, y0, mode=0):
    def insert_(img, mask, x0, y0):
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                img[x0 + x, y0 + y] = mask[x, y]
        return img

    i_h, i_w, _ = img.shape
    m_h, m_w, _ = mask.shape

    # 左上
    if mode == 0:
        if x0 - m_h >= 0 and y0 - m_w >= 0:
            return insert_(img, mask, x0 - m_h, y0 - m_w)

    # 右上
    if mode == 1:
        if x0 - m_h >= 0 and y0 + m_w < i_w:
            return insert_(img, mask, x0 - m_h, y0)

    # 右下
    if mode == 2:
        if x0 + m_h < i_h and y0 + m_w < i_w:
            return insert_(img, mask, x0, y0)

    # 左下
    if mode == 3:
        if x0 + m_h < i_h and y0 - m_w >= 0:
            return insert_(img, mask, x0, y0 - m_w)

    return None


def s_set_add(img, mask, s_set, x0, y0, mode=0):
    def s_set_add_(mask, x0, y0, s_set):
        m_h, m_w, _ = mask.shape
        for x in range(x0, x0 + m_h):
            for y in range(y0, y0 + m_w):
                s_set.add((x, y))
        return s_set

    i_h, i_w, _ = img.shape
    m_h, m_w, _ = mask.shape

    if mode == 0:
        if x0 - m_h >= 0 and y0 - m_w >= 0:
            return s_set_add_(mask, x0 - m_h, y0 - m_w, s_set)

    if mode == 1:
        if x0 - m_h >= 0 and y0 + m_w < i_w:
            return s_set_add_(mask, x0 - m_h, y0, s_set)

    if mode == 2:
        if x0 + m_h < i_h and y0 + m_w < i_w:
            return s_set_add_(mask, x0, y0, s_set)

    if mode == 3:
        if x0 + m_h < i_h and y0 - m_w >= 0:
            return s_set_add_(mask, x0, y0 - m_w, s_set)

    return s_set


def insert_video_title(img, vt, s_set, save_path=""):
    # 插入到左上角
    img_vt = insert(img, vt, 0, 0, 2)

    if img_vt is not None:
        s_set = s_set_add(img, vt, s_set, 0, 0, 2)
        return img_vt, s_set

    return img, s_set


def insert_name_card(img, nc, s_set, x_min, x_max, y_min, y_max):
    # 从右上开始，逆时针的方式尝试插入
    x_y_modes = [
        [x_min, y_max, [0, 1, 2]],
        [x_max, y_max, [1, 2, 3]],
        [x_max, y_min, [2, 3, 0]],
        [x_min, y_min, [3, 0, 1]]
    ]

    for x0, y0, modes in x_y_modes:
        for mode in modes:
            img_nc = insert(img, nc, x0, y0, mode)
            if img_nc is not None:
                s_set = s_set_add(img, nc, s_set, x0, y0, mode)
                return img_nc, s_set

    return img, s_set


def insert_comment(img, comment, s_set, last_comment):
    def get_comment_set(comment, x0, y0):
        c_h, c_w, _ = comment.shape

        c_set = set()
        if c_w > 10 and c_h > 10:
            for x in range(10, c_h, 10):
                for y in range(10, c_w, 10):
                    c_set.add((x0 + x, y0 + y))
        c_set.add((x0, y0))
        c_set.add((x0, y0 + c_w))
        c_set.add((x0 + c_h, y0))
        c_set.add((x0 + c_h, y0 + c_w))
        c_set.add((x0 + c_h // 2, y0 + c_w // 2))

        return c_set

    i_h, i_w, _ = img.shape
    c_h, c_w, _ = comment.shape

    if last_comment != (-1, -1):
        x0, y0 = last_comment
        c_set = get_comment_set(comment, x0, y0)
        if not c_set & s_set:
            img_comment = insert(img, comment, x0, y0, 2)
            return img_comment, s_set, (x0, y0)

    while True:
        x0, y0 = randint(0, i_h), randint(0, i_w)
        c_set = get_comment_set(comment, x0, y0)
        if not c_set & s_set:
            img_comment = insert(img, comment, x0, y0, 2)
            if img_comment is not None:
                s_set = s_set_add(img, comment, s_set, x0, y0, 2)
                return img_comment, s_set, (x0, y0)

    return img, s_set, (-1, -1)


def insert_all(img_path, sod_path, vt_path, nc_path, comment_path, res_path, last_comment):
    img = cv.imread(img_path)
    sod = cv.imread(sod_path)
    vt = cv.imread(vt_path)
    nc = cv.imread(nc_path)
    comment = cv.imread(comment_path)

    s_set, x_min, x_max, y_min, y_max = get_salient_set(sod)

    img, s_set = insert_video_title(img, vt, s_set)

    img, s_set = insert_name_card(img, nc, s_set, x_min, x_max, y_min, y_max)

    img, s_set, last_comment = insert_comment(img, comment, s_set, last_comment)

    cv.imwrite(res_path, img)

    return last_comment


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

    for input_video_name in os.listdir(input_video_root):
        print("===========[ start prcessing video: {} ]===========".format(input_video_name))

        res_path = os.path.join(root, 'res', input_video_name.split('.')[0])
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

        video2img(input_video_root, input_image_root)

        img2sod(root, input_image_root)

        '''insert [video title, name card, comment]'''

        sod_image_names = os.listdir(sod_input_image_root)
        sod_image_names.sort(key=lambda x: x)

        insert_all(input_image_root, output_image_root, vt_path, nc_path, comment_path)

        break

        # '''img2video'''
        # video_path_ = os.path.join(video_path, video_name)
        # recover_video_path = os.path.join(recover_path, video_name)
        # print(video_path_, recover_video_path)
        # img2gif(res_path, video_path_, recover_video_path)


if __name__ == "__main__":
    main()
