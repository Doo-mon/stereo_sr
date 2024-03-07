import os
from PIL import Image
import numpy as np


"""
 Middlebury 2014 数据集分辨率比较大 为了和Flickr1024数据集对应 首先对其进行2倍下采样处理成hr图片
"""


def modcrop(image, scale):
    size = np.array(image.size)
    size = size - size % scale
    image = image.crop((0, 0, size[0], size[1]))
    return image

def get_lr_image_2x(img_path_0, scale=2):
    img_0 = Image.open(img_path_0)
    img_hr_0 = modcrop(img_0, scale) 
    img_lr_0 = img_hr_0.resize((img_hr_0.size[0] // scale, img_hr_0.size[1] // scale), Image.BICUBIC)
    return img_lr_0


if __name__=="__main__":

    data_dir = "./datasets"

    data_dir = os.path.join(data_dir, "Middlebury_train_im0_im1")

    all_items = os.listdir(data_dir)
    folders = [item for item in all_items if os.path.isdir(os.path.join(data_dir, item))]

    for folder in folders:
        l_data_dir = os.path.join(data_dir, folder, 'im0.png')
        r_data_dir = os.path.join(data_dir, folder, 'im1.png')

        img_lr_0 = get_lr_image_2x(l_data_dir)
        img_lr_1 = get_lr_image_2x(r_data_dir)

        Image.fromarray(np.uint8(img_lr_0)).save(l_data_dir)
        Image.fromarray(np.uint8(img_lr_1)).save(r_data_dir)


