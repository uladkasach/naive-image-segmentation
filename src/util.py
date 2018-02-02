import numpy as np;


def map_image_labels_to_prediction(image, labels):
    violet = np.array([138,43,226]);
    sky = np.array([0,191,255]);
    forest = np.array([34,139,34]);
    sun = np.array([255,255,0]);
    firebrick = np.array([178,34,34]);
    lookup = [sky, forest, violet, sun, firebrick];

    print 'mapping pixels to clusters to render image...'
    prediction = np.array(image); ## create a copy
    for index, label in enumerate(labels):
        prediction[index] = lookup[label];
    return prediction;
