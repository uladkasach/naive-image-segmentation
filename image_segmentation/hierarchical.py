## http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
## http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
from sklearn.cluster import AgglomerativeClustering;
import numpy as np;
import images.loader as images;
import util;
import sys;


print sys.argv
if(len(sys.argv) < 2):
    print "error. camel number is not defined.";
    exit();
camel_number = sys.argv[1];


image, size = images.load("images/source/camel_"+camel_number+".jpg");
image = images.unravel(image, size);
print 'image[:5] : ';
print image[:5];


clustering = AgglomerativeClustering(n_clusters=3)
labels = clustering.fit_predict(image)


print type(labels);
print cluster_centers;
print labels[:5];

prediction = util.map_image_labels_to_prediction(image, labels);

print prediction[:5];


prediction = images.reravel(prediction, size);
images.save(prediction, "images/pred/hierarchical/camel_"+camel_number+".jpg");
