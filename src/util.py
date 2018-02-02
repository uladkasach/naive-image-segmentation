import numpy as np;
import matplotlib.pyplot as plt;
import json;


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

def name_gen(pred_method, camel_number, blur_number):
    camel_number = str(camel_number);
    blur_number = str(blur_number);

    blur_modifier = "";
    if(int(blur_number) > 0): blur_modifier = ".blur_"+blur_number; ## if blur_number is 0 dont add a blur modifier

    source_path = "images/source/camel_"+camel_number+blur_modifier+".jpg";
    label_path = "images/labels/camel_"+camel_number+".jpg";
    output_path = "images/pred/"+pred_method+"/camel_"+camel_number+blur_modifier+".jpg";
    stats_path = "stats/"+pred_method+"/camel_"+camel_number+blur_modifier+".pkl"
    return source_path, label_path, output_path, stats_path;

def gen_roc_stats(stats):
    roc_stats = []; ## tuples of form (fpr, tpr)
                    ## tpr = TP/T = TP/(FN + TP)
                    ## fpr = FP/N = FP/(FP + TN)
    for stat in stats:
        tpr = stat["tp"] / float(stat["fn"] + stat["tp"]);
        fpr = stat["fp"] / float(stat["fp"] + stat["tn"]);
        roc_stats.append((fpr, tpr));
    return roc_stats;

def record_roc(stats, cluster):
    plt.gcf().clear()
    print(stats);
    for key, these_stats in stats.iteritems():
        x = [s[0] for s in these_stats];
        y = [s[1] for s in these_stats];
        print(x);
        print(y);
        plt.plot(x, y, label = key);
    plt.legend()
    #plt.show()
    plt.savefig('output/cluster_' + cluster + '_roc.png');

def record_stats(stats, cluster):
    with open('output/stats.txt', "a+") as file:
        file.write("\n\nstats for cluster " + cluster + " ------\n");
        file.write(json.dumps(stats, indent=2, sort_keys=True));
