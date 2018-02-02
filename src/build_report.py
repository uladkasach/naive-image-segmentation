'''
    this file:
        1. segments the original image for each segmenting method - this is done for 5 blur ranks
            - if the result already exists, it does not rerun it
        2. calculates statistics for each blur rank and segmentation method
            - if the results already exist, it gets them from cache
        3. builds roc curve for each cluster
'''
import evaluate;
import kmeans;
import manual_threshold as manual;

source_path = "images/source/camel_"+camel_number+".jpg";
output_path = "images/pred/kmeans/camel_"+camel_number+".jpg";

kmeans.segment(source_path, output_path);
