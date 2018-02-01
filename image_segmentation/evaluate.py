import images.loader as images;
import numpy as np;
import itertools;
import sys;

def evaluate_stats_for_class(this_class, labels, predictions, mapping):
    stats = dict({
        "tp" : 0,
        "tn" : 0,
        "fp" : 0,
        "fn" : 0,
        "len" : len(labels),
    })
    for index, actual in enumerate(labels):
        pred = predictions[index];
        pred = mapping[pred];
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
    stats = [stats_0, stats_1, stats_2];

    ## recall and precision defined in https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co

    acc = (stats_0["tp"] + stats_1["tp"] + stats_2["tp"])/float(len(labels));
    return acc, stats;


def colors_to_clusters(image): ## convert each rgb pixel color to a unique id (i.e., cluster id)
    map = dict();
    clusters = [];
    whitelist = ["255255255", "124124124", "000", "13843225", "3413934", "0191254"] ## predefined list of valid keys - in images the areas between where they change is not kept static
    for index, pixel in enumerate(image):
        key = ''.join(str(e) for e in pixel)  ## converts pixel rgb value to string
        #print(pixel);
        if(key not in whitelist):
            clusters.append(-1); ## distinguish invalid pixels so we can remove from both label and pred list later
            continue;
        if(key not in map): map[key] = len(map); ## assign a unique id
        clusters.append(map[key]);

    print(map);
    return clusters;

def normalize_to_well_defined(labels, predictions):
    final_label = [];
    final_pred = [];
    for index, label in enumerate(labels):
        pred = predictions[index];
        if(pred == -1 or label == -1): continue;
        final_label.append(label);
        final_pred.append(pred);
    return final_label, final_pred;


def evaluate_strategy_and_picture(strategy, camel_number):
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

    print("normalizing to include only equally well defined pixels...");
    label, pred = normalize_to_well_defined(label, pred);

    print("finding min-error clusters...");
    ## we need to find the pred cluster id that, when assumed as correlating to the label cluster id, minimizes the error.
    ## we have 3 label clusters and 3 pred clusters. This is a permutation problem: 3*2*1 -> 6 different permutations possible
    ## i.e., [0, 1, 2], [1, 2, 0], etc
    permutations = list(itertools.permutations([0, 1, 2]));
    max_acc = 0;
    for perm in permutations:
        mapping = perm; ## i.e., label id 0 maps to pred id mapping[0]
        acc, stats = evaluate_stats(mapping, label, pred)
        if(acc > max_acc):
            max_acc = acc;
            best_stats = stats;
        print("accuracy for permutation " + str(perm) + " : " + str(acc));
    print("best results : " + str(best_stats));
    return best_stats;

def merge_stats(prev, new):
    if(prev == False): prev = dict({"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0});
    return dict({
        "tp" : prev["tp"] + new["tp"]/float(new["len"]), ## divide by len to normalize each value, independent of pixel count
        "tn" : prev["tn"] + new["tn"]/float(new["len"]),
        "fp" : prev["fp"] + new["fp"]/float(new["len"]),
        "fn" : prev["fn"] + new["fn"]/float(new["len"]),
    });
def evaluate(strategy):
    assert strategy in ["kmeans", "manual_threshold"]; ## ensure that the strategy is from one of the well defined ones
    print("----------- begining evaluation of " + strategy + " -----------")
    by_class_stats = dict({
        "0" : False,
        "1" : False,
        "2" : False,
    });

    ## calculate statistics
    for index, camel in enumerate([1, 2, 3, 4, 8, 10]):
        stats_for_image = evaluate_strategy_and_picture(strategy, camel);
        by_class_stats["0"] = merge_stats(by_class_stats["0"], stats_for_image[0])
        by_class_stats["1"] = merge_stats(by_class_stats["1"], stats_for_image[1])
        by_class_stats["2"] = merge_stats(by_class_stats["2"], stats_for_image[2])

    print("total results :");
    print(by_class_stats);



if __name__ == "__main__":
    print(sys.argv)
    kmeans = evaluate("kmeans");
    #kmeans = evaluate("kmeans");
    #manual = evaluate("manual_threshold");
