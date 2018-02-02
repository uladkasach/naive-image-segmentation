## note, feature space is spanned by 3 dimensions: (r, g, b). not by spacial image dimensions, (x, y).
## i.e., image = list of data points which we want to cluster

from sklearn.cluster import KMeans;
import numpy as np;
import images.loader as images;
import util;
import sys;
import os.path



def segment(source_path, output_path):
    print("");
    print("------ segmenting with kmeans on " + source_path + " ------")

    if(os.path.isfile(output_path)):
        print(output_path + " already exists");
        return True;

    image, size = images.load(source_path);
    image = images.unravel(image, size);
    print 'image[:5] : ';
    print image[:5];

    if(False): ## to test the reravel operation
        image = images.reravel(image, size);
        images.save(image, 'test.jpg');
        exit();

    print 'running k means...'
    kmeans = KMeans(n_clusters=3, random_state=0).fit(image)
    print kmeans.labels_[:10];
    print kmeans.cluster_centers_
    labels = kmeans.labels_;


    prediction = util.map_image_labels_to_prediction(image, labels);

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
    output_path = "images/pred/kmeans/camel_"+camel_number+".jpg";
    segment(source_path, output_path);
