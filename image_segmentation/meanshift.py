## http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift
## http://scikit-learn.org/stable/modules/clustering.html#mean-shift


from sklearn.cluster import MeanShift, estimate_bandwidth;
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


# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(image, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(image)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
assert n_clusters_ < 6

print type(labels);
print cluster_centers;
print labels[:5];

prediction = util.map_image_labels_to_prediction(image, labels);

print prediction[:5];

modifier = "";
if(n_clusters_ != 3) modifier = "not_3_clusters/";
prediction = images.reravel(prediction, size);
images.save(prediction, "images/pred/meanshift/"+modifier+"camel_"+camel_number+".jpg");
