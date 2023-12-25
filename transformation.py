from sklearn.base import BaseEstimator, TransformerMixin
import skimage
import numpy as np
from skimage.feature import hog
from skimage import exposure
from skimage.feature import canny
from skimage.transform import rescale
from skimage.filters import gaussian
from skimage.segmentation import slic
from skimage.filters import gabor
from scipy import ndimage as ndi


class Scale(BaseEstimator, TransformerMixin):

    def __init__(self, size):
        self.size = size

    def transform(self, images):
        return [skimage.transform.resize(img, self.size) for img in images]


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


class Gray(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        gray_images = []
        for img in X:
            if img.shape[2] == 4:
                img = skimage.color.rgba2rgb(img)
            gray_img = skimage.color.rgb2gray(img)
            gray_images.append(gray_img)
        return np.array(gray_images)


class HistogramEqualization(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        equalized_images = [exposure.equalize_hist(img) for img in X]
        return np.array(equalized_images)


class EdgeDetection(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        edge_images = [canny(img) for img in X]
        return np.array(edge_images)


class GaussianBlur(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def transform(self, X, y=None):
        blurred_images = [gaussian(img, sigma=self.sigma) for img in X]
        return np.array(blurred_images)


class ImageNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, range=(0, 1)):
        self.range = range

    def transform(self, X, y=None):
        min_val, max_val = self.range
        normalized_images = [(img - img.min()) / (img.max() - img.min()) * (max_val - min_val) + min_val for img in X]
        return np.array(normalized_images)


class SuperResolution(BaseEstimator, TransformerMixin):
    def __init__(self, scale_factor=2.0):
        self.scale_factor = scale_factor

    def transform(self, X, y=None):
        super_res_images = [rescale(img, self.scale_factor, mode='reflect', multichannel=True) for img in X]
        return np.array(super_res_images)


class ImageSegmentation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        segmented_images = [slic(img, compactness=10, n_segments=100) for img in X]
        return np.array(segmented_images)


def extract_gabor_features(image, kernels):
    features = []
    for kernel in kernels:
        filtered_image = ndi.convolve(image, kernel, mode='wrap')
        features.append(np.mean(filtered_image))
        features.append(np.var(filtered_image))
    return features


class GaborFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=8, scales=[0.1, 0.5, 1, 2]):
        self.orientations = orientations
        self.scales = scales
        self.kernels = self._generate_gabor_kernels()

    def _generate_gabor_kernels(self):
        return [
            (gabor(frequency=0.6, theta=theta, sigma_x=sigma, sigma_y=sigma), theta, sigma)
            for theta in np.linspace(0, np.pi, self.orientations, endpoint=False)
            for sigma in self.scales
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        gabor_features = [self._apply_gabor_filters(img) for img in X]
        return np.array(gabor_features)

    def _apply_gabor_filters(self, img):
        return [self._compute_gabor_response(img, kernel, theta, sigma) for kernel, theta, sigma in self.kernels]

    def _compute_gabor_response(self, img, kernel, theta, sigma):
        # Apply Gabor filter to the image
        filtered_img = gabor(img, frequency=0.6, theta=theta, sigma_x=sigma, sigma_y=sigma, mode='wrap')
        # Compute magnitude of response
        response_magnitude = np.sqrt(filtered_img.real**2 + filtered_img.imag**2)
        return response_magnitude


class ImageFlattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Flatten each image in the dataset
        return np.array([image.flatten() for image in X])