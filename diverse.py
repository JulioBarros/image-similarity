import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import scipy



def calculate_distances(features):
    return scipy.spatial.distance.cdist(features, features, "cosine")

features = pickle.load(open("public/features.pickle","rb"))

filenames = open('public/image_file_names.txt','rt').readlines()

pca=PCA(n_components=10)
pca.fit(features)
X_pca=pca.transform(features)


f1 = [filenames[i] for i in X_pca.argmax(axis=0)]

f2 = [filenames [i] for i in X_pca.argmin(axis=0)]

diverse = f1 + f2

print([os.path.basename(f.strip()) for f in diverse])
