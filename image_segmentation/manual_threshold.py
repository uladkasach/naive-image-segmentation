## http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
## http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
from sklearn.cluster import AgglomerativeClustering;
import numpy as np;
import images.loader as images;
import util;
import sys;
import os.path



def segment(source_path, output_path):
    print("");
    print("------ segmenting with manual threshold on " + source_path + " ------")
    if(os.path.isfile(output_path)):
        print(output_path + " already exists");
        return True;

    threshold = [255/float(3), 255*2/float(3)]

    print sys.argv
    if(len(sys.argv) < 2):
        print "error. camel number is not defined.";
        exit();
    camel_number = sys.argv[1];


    image, size = images.load(source_path);
    image = images.unravel(image, size);
    print 'image[:5] : ';
    print image[:5];


    def rgb2gray(rgb):
        r, g, b = rgb[0], rgb[1], rgb[2];
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    print ("applying manual thresholds...")
    labels = [];
    for pixel in image:
        # print pixel;
        gray = rgb2gray(pixel);
        if(gray < threshold[0]):
            label = 0;
        elif(gray < threshold[1]):
            label = 1;
        else:
            label = 2;
        labels.append(label);

    print type(labels);
    print labels[:5];

    prediction = util.map_image_labels_to_prediction(image, labels);
    print prediction[:5];

    prediction = images.reravel(prediction, size);
    images.save(prediction, output_path);
    return True;

if __name__ == "__main__":
    print sys.argv
    if(len(sys.argv) < 2):
        print "error. camel number is not defined.";
        exit();
    camel_number = sys.argv[1];

    source_path = "images/source/camel_"+camel_number+".jpg";
    output_path = "images/pred/manual_threshold/camel_"+camel_number+".jpg";
    segment(source_path, output_path);
