import argparse
import os
import pathlib

import numpy as np
import keras
import keras.applications as kapp
from PIL import Image, ExifTags
from PIL import Image as pil_image
import random
import scipy
import pickle
import json
    
from collections import namedtuple
from jinja2 import Environment, FileSystemLoader


# Define a named tuple to keep our image information together
ImageFile = namedtuple("ImageFile", "src filename path uri")


# Define a new filter for jinja to get the name of the file from a path
def basename(text):
    return text.split(os.path.sep)[-1]


env = Environment(loader=FileSystemLoader("./templates"))
env.filters["basename"] = basename


def is_image(filename):
    """ Checks the extension of the file to judge if it an image or not. """
    fn = filename.lower()
    return fn.endswith("jpg") or fn.endswith("jpeg") or fn.endswith("png")


def find_image_files(root):
    """ Starting at the root, look in all the subdirectories for image files. """
    file_names = []
    for path, _, files in os.walk(os.path.expanduser(root)):
        for name in files:
            if is_image(name):
                file_names.append(os.path.join(path, name))
    return file_names


def build_model(model_name):
    """ Create a pretrained model without the final classification layer. """
    if model_name == "resnet50":
        model = kapp.resnet50.ResNet50(weights="imagenet", include_top=False)
        return model, kapp.resnet50.preprocess_input
    elif model_name == "vgg16":
        model = kapp.vgg16.VGG16(weights="imagenet", include_top=False)
        return model, kapp.vgg16.preprocess_input
    elif model_name == "vgg16block4":
        base_model = kapp.vgg16.VGG16(weights='imagenet')
        model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        return model, kapp.vgg16.preprocess_input
    else:
        raise Exception("Unsupported model error")


def fix_orientation(image):
    """ Look in the EXIF headers to see if this image should be rotated. """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        return image
    except (AttributeError, KeyError, IndexError):
        return image


def extract_center(image):
    """ Most of the models need a small square image. Extract it from the center of our image."""
    width, height = image.size
    new_width = new_height = min(width, height)

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))


def process_images(file_names, preprocess_fn):
    """ Take a list of image filenames, load the images, rotate and extract the centers, 
    process the data and return an array with image data. """
    image_size = 224
    print_interval = len(file_names) / 10

    image_data = np.ndarray(shape=(len(file_names), image_size, image_size, 3))
    for i, ifn in enumerate(file_names):
        im = Image.open(ifn.src)
        im = fix_orientation(im)
#        im = extract_center(im)
        im = im.resize((image_size, image_size)) #,pil_image.NEAREST)
        im = im.convert(mode="RGB")
        filename = os.path.join(ifn.path, ifn.filename + ".jpg")
        im.save(filename)
        image_data[i] = np.array(im)
        if i % print_interval == 0:
            print("Processing image:", i, "of", len(file_names))
    return preprocess_fn(image_data)


def generate_features(model, images):
    return model.predict(images)


def calculate_distances(features):
    return scipy.spatial.distance.cdist(features, features, "cosine")


def generate_site(output_path, names, features):
    """ Take the features and image information. Find the closest features and
    and generate static html files with similar images."""

    template = env.get_template("image.html")
    print_interval = len(names) / 10

    # Calculate all pairwise distances
    print("Calculating distances")
    distances = calculate_distances(features)

    print("Generating static pages")
    results = []
    # Go through each image, sort the distances and generate the html file
    for idx, ifn in enumerate(names):
        output_filename = os.path.join(output_path, ifn.filename + ".html")
        dist = distances[idx]
        similar_image_indexes = np.argsort(dist)[:13]
        dissimilar_image_indexes = np.argsort(dist)[-12:]
        similar_images = [names[i] for i in similar_image_indexes if i != idx]
        dissimilar_images = [names[i] for i in dissimilar_image_indexes if i != idx]
        html = template.render(
            target_image=ifn,
            similar_images=similar_images,
            dissimilar_images=dissimilar_images,
        )
        with open(output_filename, "w") as text_file:
            text_file.write(html)
        if idx % print_interval == 0:
            print("Generating page:", idx, "of", len(names))

        results.append({"filename" : ifn.filename, "most" : similar_images, "least" : dissimilar_images })

    template = env.get_template("index.html")
    with open(os.path.join(output_path, "index.html"), "w") as text_file:
        text_file.write(template.render(num_images=len(names)))

    return results


def parse_args():
    """Set up the various command line parameters."""

    parser = argparse.ArgumentParser(description="Image similarity")

    parser.add_argument(
        "-i",
        "--inputdir",
        help="The directory to search for images",
        default="~/Pictures",
    )

    parser.add_argument(
        "-o",
        "--outputdir",
        help="The directory to write output html files",
        default="./public",
    )

    parser.add_argument(
        "-n",
        "--num_images",
        help="The maximum number of images to process. If more images are found n of them will be chosen randomly.",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="The model to use to extract features.",
        choices=["resnet50", "vgg16", "vgg16block4"],
        default="vgg16",
    )

    return parser.parse_args()


def ensure_directory(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    parser = parse_args()
    input_dir = parser.inputdir
    output_dir = parser.outputdir
    image_output_dir = os.path.join(output_dir, "images")
    num_images = parser.num_images

    ensure_directory(image_output_dir)
    model, preprocess_fn = build_model(parser.model)

    # Find the image files and if we want to use less than are available pick them randomly
    image_file_names = find_image_files(input_dir)
    n = len(image_file_names)

    if n > num_images:
        n = num_images
        # image_file_names = random.sample(image_file_names, n)
        image_file_names = image_file_names[:n]

    print(f"Using {n} images from {input_dir}")
    with open(os.path.join(output_dir, 'image_file_names.txt'), 'w') as f:
        for ifn in image_file_names:
            f.write("%s\n" % ifn)

    # namedtuple('ImageFile', 'src filename path uri')
    image_files = [
        ImageFile(src_file, str(i), image_output_dir, "images/" + str(i) + ".jpg")
        for i, src_file in enumerate(image_file_names)
    ]


    image_data = process_images(image_files, preprocess_fn)
    print("Image data shape:", image_data.shape)

    # create the model and run predict on the image data to generate the features
    print("Generating features for images")
    features = generate_features(model, image_data)
    features = features.reshape(features.shape[0], -1)
    print("features shape:", features.shape)
    with open(os.path.join(output_dir, 'features.pickle'), 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # generate the static pages using the images and feature matrix
    matches = generate_site(output_dir, image_files, features)
    print("Writing out features pickle file")
    with open(os.path.join(output_dir, 'matches.json'), 'w') as outfile:
        json.dump(matches, outfile)
        
    print("Done")


if __name__ == "__main__":
    main()
