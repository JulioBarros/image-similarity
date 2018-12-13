# Image Similarity Demo

This is the start of a demo on how to use a deep neural network to generate features that can be used to gauge the similarity of images.

This works as is and is currently a work in progress. I hope to add a host of small improvements in the near future.

## Running the script

After installing the packeges in `requirements.txt` you should be able to run the script

```
python image_similarity.py -i <your image directory>
```

The script searches for all images (by extension) from the image directory, calculates their features, for each image it calculates the other most similar images and generates an html page which is written to the `public` directory.

You can then open public/index.html or start a simple python server `python -m http.server` from that directory to browse the images.

## How it works

The key is to use a pretrained network and get the features from the output of a layer before the final input layer. Right now we use ResNet50, for no particular reason, trained on ImageNet. Other models and layers may give different results.

We find each image in the input directory, resize it for the model and then run predict on it to generate features.

When we want to find simliar images to a particular image we use a similarity metric between that image's feature vector and the other known features and find the smallest 'distances'. Currently we use cosine distance, though we could try others to see how they work.

## Future improvements

Things I may add in the future

- ability to choose the max number of images to use
- ability to specify output directory
- ability to specify different models to use
- ability to specify distance metric
- automatically finding (near) duplicates
- anything else I or you may think of
