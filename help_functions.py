import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
import skimage
from skimage import exposure
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from skimage.filters import gabor

def read_image_from_csv(csv_file):
    images = []
    data = pd.read_csv(csv_file)

    for index, row in data.iterrows():
        img_string = row['image_data']
        img_data = np.array([int(pixel) for pixel in img_string.split(',')])
        img_shape = tuple(map(int, row['shape'][1:-1].split(',')))
        img_data = img_data.reshape(img_shape)
        img = Image.fromarray(img_data.astype(np.uint8))
        images.append(img)

    return images

def read_labels(csv_file):
    labels = []
    data = pd.read_csv(csv_file)

    for index, row in data.iterrows():
        label = row['type']
        labels.append(label)

    return labels

types_names = ["artilery",
               "bmp",
               "mlrs",
               "mrlb",
               "spg",
               "tank",
               "track"]

labels = np.unique(types_names)


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

class NormalizationTransformer(BaseEstimator, TransformerMixin):
    """
    Normalize the input data using StandardScaler
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """Fit the StandardScaler on the input data"""
        X_flattened = np.vstack(X)
        self.scaler.fit(X_flattened)
        self.num_features_ = X_flattened.shape[1]
        return self

    def transform(self, X, y=None):
        """Normalize the input data using StandardScaler"""
        X_flattened = np.vstack(X)
        X_normalized = self.scaler.transform(X_flattened)
        return [X_normalized[i * self.num_features_:(i + 1) * self.num_features_] for i in range(len(X))]


class ColorHistogramTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        histograms = []
        for image in X:
            hist_r, _ = exposure.histogram(image[:, :, 0].flatten())
            hist_g, _ = exposure.histogram(image[:, :, 1].flatten())
            hist_b, _ = exposure.histogram(image[:, :, 2].flatten())
            histograms.append(np.concatenate([hist_r, hist_g, hist_b]))
        return np.array(histograms)

class GaborTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for image in X:
            _, feature = gabor(image, frequency=0.6)
            features.append(feature.flatten())
        return np.array(features)
