'''
    this file:
        1. segments the original image for each segmenting method - this is done for 5 blur ranks
            - if the result already exists, it does not rerun it
        2. calculates statistics for each blur rank and segmentation method
            - if the results already exist, it gets them from cache
        3. builds roc curve for each cluster
'''
import evaluate as evaluate;
import kmeans;
import manual_threshold as manual;
import util;

pred_methods = dict({"kmeans":kmeans, "manual":manual})
segmentation_options = list(pred_methods.keys());
camel_options = [1, 2, 3, 4, 8, 10]
blur_options = [0, 1, 2, 3, 4, 5]

def segment(pred_method, camel_number, blur_number):
    #print("segmenting " + pred_method + " for camel " + camel_number + " and blur " + blur_number);
    source_path, __, output_path, __ = util.name_gen(pred_method, camel_number, blur_number);
    pred_methods[pred_method].segment(source_path, output_path);


## 1 segment each (segmentation_method, blur_number, camel_number)
for segment_method in segmentation_options:
    for camel in camel_options:
        for blur in blur_options:
            segment(segment_method, camel, blur);

## 2 for each (seg_meth, blur) calculate the aggragate statistics
compiled_stats = dict({
    "0" : {
        "kmeans" : [],
        "manual" : [],
    },
    "1" : {
        "kmeans" : [],
        "manual" : [],
    },
    "2" : {
        "kmeans" : [],
        "manual" : [],
    }
})
roc_stats = dict({"0":{}, "1":{},"2":{}});
for segment_method in segmentation_options:
    for blur in blur_options:
        results = evaluate.evaluate(segment_method, blur);
        compiled_stats["0"][segment_method].append(results["0"]);
        compiled_stats["1"][segment_method].append(results["1"]);
        compiled_stats["2"][segment_method].append(results["2"]);
    roc_stats["0"][segment_method] = util.gen_roc_stats(compiled_stats["0"][segment_method])
    roc_stats["1"][segment_method] = util.gen_roc_stats(compiled_stats["1"][segment_method])
    roc_stats["2"][segment_method] = util.gen_roc_stats(compiled_stats["2"][segment_method])

## 3 build roc curves for each cluster and report rates for each cluster and blur
print("");
print("RECORDING OUTPUT...");
for cluster in ["0", "1", "2"]:
    util.record_roc(roc_stats[cluster], cluster)
    util.record_stats(compiled_stats[cluster], cluster);
