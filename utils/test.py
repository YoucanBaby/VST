import os
from random_insert import insert

root_path = '../preds/XYF/'
dataset_path = root_path + 'image/'
comment_path = root_path + 'comment.png'
save_path = root_path + 'comment/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for img_name in os.listdir(dataset_path):
    if not img_name.split('.')[0].isdigit():
        continue
    img_path = dataset_path + img_name
    save_img_path = save_path + img_name
    random_insert(img_path, comment_path, save_img_path)

