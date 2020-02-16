
import pickle
import warnings
import datetime
import optparse
import os, errno
from keras.preprocessing.image import array_to_img
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).

    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

def read_train_data_multimodal(data_file, delim="\t"):
    """
    Prepare the data
    """
    image_list = []
    with open(data_file, 'rU') as f:
        #next(f)
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            #txt = row[2].strip()
            image_path = row[0].strip()
            image_list.append(image_path)

    return image_list;

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="input_file_name", default=None, type="string")
    parser.add_option('-o', action="store", dest="output_file_name", default=None, type="string")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    input_file_name = options.input_file_name
    output_file_name = options.output_file_name
    data = read_train_data_multimodal(input_file_name)

    images_npy_data = {}  # np.empty([len(instances), 224, 224, 3])
    for i ,img_path in enumerate(data):
        #img_path = inst.imgpath
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        images_npy_data[img_path] = img_data

    with open(output_file_name, 'wb') as handle:
        pickle.dump(images_npy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #with open('../filename.pickle', 'rb') as handle:
    #    images_npy_data = pickle.load(handle)
