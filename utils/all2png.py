import cv2 as cv
import os 


"""把数据集中的图像都转为.png
"""
# def main():  
#     datasets_path = '/home/xuyifang/dataset/SOD'
#     for dataset in os.listdir(datasets_path):

#         dataset_path = os.path.join(datasets_path, dataset)
#         for image_gt in os.listdir(dataset_path):

#             image_gt_path = os.path.join(datasets_path, dataset, image_gt)
#             for image_name in os.listdir(image_gt_path):
#                 if '.jpg' in image_name:
#                     try:
#                         img = cv.imread(image_gt_path + '/' + image_name)

#                         save_path = '/home/xuyifang/dataset/SOD_'
#                         save_path = os.path.join(save_path, dataset, image_gt, image_name.split('.')[0] + '.png')

#                         cv.imwrite(save_path, img)
#                     except:
#                         print(image_gt_path + '/' + image_name)
#                         print(save_path)

#         print(dataset_path)


"""把SOD数据集的名称都补充为6位
"""
# def main():
#     dataset_path = '/home/xuyifang/dataset/SOD/SOD/image/'
#     for img_name in os.listdir(dataset_path):
#         img_path = dataset_path + img_name
#         img = cv.imread(img_path)

#         img_num = img_name.split('.')[0]
#         img_num = img_num.zfill(6)
#         img_name = img_num + '.png'

#         img_path = dataset_path + img_name
#         cv.imwrite(img_path, img)


def main():
    dataset_path = '/home/xuyifang/dataset/SOD/XYF/image/'
    
    for img_name in os.listdir(dataset_path):
        img_path = dataset_path + img_name
        img = cv.imread(img_path)
        cv.imwrite(img_path, img)


if __name__ == '__main__':
    main()

