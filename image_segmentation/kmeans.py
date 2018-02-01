## note, feature space is spanned by 3 dimensions: (r, g, b). not by spacial image dimensions, (x, y).
## i.e., image = list of data points which we want to cluster

from sklearn.cluster import KMeans;
import numpy as np;
import images.loader as images;
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

if(False): ## to test the reravel operation
    image = images.reravel(image, size);
    images.save(image, 'test.jpg');
    exit();

print 'running k means...'
kmeans = KMeans(n_clusters=3, random_state=0).fit(image)
print kmeans.labels_[:10];
print kmeans.cluster_centers_
labels = kmeans.labels_;

violet = np.array([138,43,226]);
sky = np.array([0,191,255]);
forest = np.array([34,139,34]);
lookup = [sky, forest, violet];

print 'mapping pixels to clusters to render image...'
prediction = np.array(image); ## create a copy
for index, label in enumerate(labels):
    prediction[index] = lookup[label];

prediction = images.reravel(prediction, size);
images.save(prediction, "images/pred/kmeans/camel_"+camel_number+".jpg");
