from help_functions import *
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import os
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder

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


data = dict()
data['label'] = []
data['filename'] = []
data['data'] = []

pklname = f"{'tank_bmp'}_{500}x{400}px.pkl"

def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        im = imread(os.path.join(folder, filename))
        im = resize(im, (256, 256))
        if im.shape[2] == 4:
            im = color.rgba2rgb(im)
        data['label'].append(split_name(filename)[0])
        data['filename'].append(filename)
        data['data'].append(im)


load_images_from_folder("tank_bmp")


types = ['tank', 'bmp', 'btr', 'mtlb', 'spg', 'mlrs']

# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])

# set up the matplotlib figure and axes, based on the number of labels
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(15, 4)
fig.tight_layout()

# make a plot for every label (equipment) type. The index method returns the
# index of the first item corresponding to its search string, label in this case
for ax, label in zip(axes, labels):
    idx = data['label'].index(label)

    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)

X = np.array(data['data'])
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)
scalify = StandardScaler()

param_grid = {
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
    'classifier__learning_rate': ['optimal', 'constant', 'invscaling'],
    'classifier__eta0': [0.01, 0.1, 1.0],
}

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)

pipeline = Pipeline([
    ('grayify', RGB2GrayTransformer()),
    ('hogify', hogify),
    ('scalify', StandardScaler()),
    ('classifier', sgd_clf),
])



grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Percentage correct: ', 100 * accuracy)
