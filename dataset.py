from PIL import Image
import numpy as np
import os
import pandas as pd
import csv

i = 0

def split_name(imageName):
    type = ""
    name = ""
    counter = 0
    for j in range(0, len(imageName)):
        if imageName[j] == "_" and counter == 0:
            counter += 1
        elif counter == 0:
            type += imageName[j]
        elif (j+8 == len(imageName)):
            break
        else:
            name += imageName[j]
    return type, name

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img_np = np.array(img)
            images.append(img_np)
    return images

images = load_images_from_folder("dataset")

def flatten_image(img):
    return ','.join(map(str, img.flatten()))

image_data_list = []
name_list = []
type_list = []
shape_list = []

i = 0
for filename, img in zip(os.listdir("dataset"), images):
    type_, name = split_name(filename)
    flattened_img = flatten_image(img)
    image_data_list.append(flattened_img)
    name_list.append(name)
    shape_list.append(images[i].shape)
    type_list.append(type_)
    i += 1
    print(filename)

df = pd.DataFrame({'name': name_list, 'image_data': image_data_list, 'type': type_list, "shape": shape_list})
df.to_csv('database1.csv', index=False)
