import argparse
import glob
import fnmatch
import os
import pathlib

import numpy as np
import keras
from keras.models import Model
from PIL import Image, ExifTags
import random
import scipy

from collections import namedtuple
from jinja2 import Environment, FileSystemLoader, select_autoescape


ImageFile = namedtuple('ImageFile', 'src filename path uri')


def basename(text):
    return text.split(os.path.sep)[-1]

env = Environment(loader=FileSystemLoader('./templates'))
env.filters['basename'] = basename


def is_image(filename):
    fn = filename.lower()
    return fn.endswith('jpg') or fn.endswith('jpeg') or fn.endswith('png')

def find_image_files(root):
    file_names = []
    for path, subdirs, files in os.walk(os.path.expanduser(root)):
        for name in files:
            if is_image(name):
                file_names.append(os.path.join(path, name))
    return file_names

def build_model():
    base_model = keras.applications.resnet50.ResNet50()
    input_layer = base_model.input
    target_layer = base_model.layers[-2]

    model = Model(inputs=input_layer,outputs=target_layer.output)
    
    return model, 2048

def fix_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        return image
    except (AttributeError, KeyError, IndexError):
        return image
    
def extract_center(image):
    width, height = image.size
    new_width = new_height = min(width,height)

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    return image.crop((left, top, right, bottom))

def process_images(file_names):
    image_size = 224
    
    image_data = np.ndarray(shape=(len(file_names),image_size,image_size,3))
    for i, ifn in enumerate(file_names):
        im = Image.open(ifn.src)
        im = fix_orientation(im)
        im = extract_center(im)
        im = im.resize((image_size,image_size))
        im = im.convert(mode='RGB')
        image_data[i] = np.array(im)
        filename = os.path.join(ifn.path, ifn.filename + ".jpg")
        im.save(filename)
        if i % 1000 == 0:
            print('Processing image:', i, 'of', len(file_names))
    return image_data

def generate_features(model, images):
    return model.predict(images)

def find_similar_images(image_features, target_image_vector,num_picks=13):
    v = target_image_vector.reshape(1, -1)
    dist = scipy.spatial.distance.cdist(image_features, v, 'cosine').reshape(-1)

    return np.argsort(dist)[:num_picks]

def generate_site(output_path, names, features):
    template = env.get_template('image.html')

    for idx, ifn in enumerate(names):
        output_filename = os.path.join(output_path,ifn.filename + '.html')
        target_image = features[idx]
        similar_image_indexes = find_similar_images(features, target_image)
        similar_images = [names[i] for i in similar_image_indexes if i != idx]
        html = template.render(target_image=ifn, similar_images=similar_images)
        with open(output_filename, "w") as text_file:
            text_file.write(html)
            
    template = env.get_template('index.html')
    with open(os.path.join(output_path,"index.html"), "w") as text_file:
            text_file.write(template.render(num_images = len(names)))


def parse_args():
    parser = argparse.ArgumentParser(description="Image similarity")
    
    parser.add_argument('-i','--inputdir',
                        help='the directory to search for images',
                        default="~/Pictures")
    
    return parser.parse_args()

def ensure_directory(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True) 

def main():
    input_dir = '~/Pictures'
    output_dir = 'public/'
    image_output_dir = os.path.join(output_dir, 'images')
    max_n = 8000

    # Find the image files and if we want to use less than are available pick them randomly
    image_file_names = find_image_files(input_dir)
    n = len(image_file_names)

    if n > max_n:
        n = max_n
        image_file_names = random.sample(image_file_names, n)

    print(f'Using {n} images from {input_dir}')
    
    # namedtuple('ImageFile', 'src filename path uri')
    image_files = [ImageFile(src_file, str(i), image_output_dir, "images/" + str(i) + ".jpg") for i, src_file in enumerate(image_file_names)]
    ensure_directory(image_output_dir)

    image_data = process_images(image_files)

    # create the model and run predict on the image data to generate the features
    print('Generating features for images')
    model, _ = build_model()
    features = generate_features(model, image_data)

    # generate the static pages using the images and feature matrix
    print('Generating static pages')
    generate_site('public', image_files, features)
    print('Done')

if __name__ == '__main__':
    main()
    
