## http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

import images.loader as images;
import numpy as np;
import itertools;

def evaluate_stats_for_class(this_class, labels, predictions, mapping):
    stats = dict({
        "tp" : 0,
        "tn" : 0,
        "fp" : 0,
        "fn" : 0,
    })
    for index, actual in enumerate(labels):
        pred = predictions[index];
        print(pred);
        print(mapping);
        mapped_pred = mapping[pred];
        if(pred == this_class and actual == this_class):
            stats["tp"] += 1;
        elif(pred != this_class and actual != this_class):
            stats["tn"] += 1;
        elif(pred == this_class and actual != this_class):
            stats["fp"] += 1;
        else:
            stats["fn"] += 1;
    return stats;


def evaluate_stats(mapping, labels, predictions):
    stats_0 = evaluate_stats_for_class(0, labels, predictions, mapping);
    stats_1 = evaluate_stats_for_class(1, labels, predictions, mapping);
    stats_2 = evaluate_stats_for_class(2, labels, predictions, mapping);

    ## recall and precision defined in https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co

    acc = (stats_0["tp"] + stats_1["tp"] + stats_2["tp"])/float(len(label));
    return acc;

def colors_to_clusters(image): ## convert each rgb pixel color to a unique id (i.e., cluster id)
    map = dict();
    clusters = [];
    whitelist = ["255255255"] ## predefined list of valid keys - in images the areas between where they change is not kept static
    for pixel in image:
        print(pixel);
        key = ''.join(str(e) for e in pixel)  ## converts pixel rgb value to string
        if(key not in whitelist): continue;
        if(key not in map): map[key] = len(map); ## assign a unique id
        clusters.append(map[key]);

    print(map);
    return clusters;


def evaluate(strategy, camel_number):
    assert strategy in ["kmeans", "manual_threshold"]; ## ensure that the strategy is from one of the well defined ones
    assert camel_number in [1, 2, 3, 4, 8, 10]; ## ensure we have a label for the camel

    ## load label and prediction image and unravel them into 'list'
    print("loading label...");
    label, size = images.load("images/labels/camel_"+str(camel_number)+".jpg");
    label = images.unravel(label, size);
    print("loading prediction...");
    pred, size = images.load("images/pred/"+strategy+"/camel_"+str(camel_number)+".jpg");
    pred = images.unravel(pred, size);

    print("extracting cluster ids...");
    label = colors_to_clusters(label);
    print(label[:5]);
    pred = colors_to_clusters(pred);
    print(pred[:5]);

    print("finding min-error clusters...");
    ## we need to find the pred cluster id that, when assumed as correlating to the label cluster id, minimizes the error.
    ## we have 3 label clusters and 3 pred clusters. This is a permutation problem: 3*2*1 -> 6 different permutations possible
    ## i.e., [0, 1, 2], [1, 2, 0], etc
    permutations = list(itertools.permutations([0, 1, 2]));
    for perm in permutations:
        mapping = perm; ## i.e., label id 0 maps to pred id mapping[0]
        acc = evaluate_stats(mapping, label, pred)
        print("accuracy for permutation " + str(perm) + " : " + str(acc));
        exit();



if __name__ == "__main__":
    evaluate("kmeans", 2);
