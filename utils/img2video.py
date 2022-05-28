import cv2 as cv
import os
import shutil


def img2video(image_path, video_path, video_name):
    vc = cv.VideoCapture('/home/v-yifangxu/Desktop/VST/' + 'preds/XYF/video/video.mp4')
    fps = vc.get(cv.CAP_PROP_FPS)
    print('fps: {}'.format(fps))

    vc_isOpened, image = vc.read()
    h, w, _ = image.shape
    size = (h, w)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    vw = cv.VideoWriter(video_path, fourcc, fps, (w, h), True)

    items = os.listdir(image_path)
    items.sort(key=lambda x: int(x.split('.')[0]))

    for item in items:
        item = image_path + item
        image = cv.imread(item)
        vw.write(image)

    vw.release()
    print('finish!')


root_path = '/home/v-yifangxu/Desktop/VST/'
# 图像存放路径
image_path = root_path + 'preds/MSRA/image/'
# 视频存放路径
video_path = root_path + 'preds/MSRA/comment/'

for video_name in os.listdir(image_path):
    image = image_path + video_name
    video = video_path

    img2video(image, video, video_name)
