# Naive Image Segmentation

This project was completed as a homework assignment for the Biometrics course at IUPUI.

This project was developed with Python v2.7.12.


![source_example](https://raw.githubusercontent.com/uladkasach/naive-image-segmentation/master/src/images/source/camel_2.jpg)
![kmeans_example](https://raw.githubusercontent.com/uladkasach/naive-image-segmentation/master/src/images/pred/kmeans/camel_2.jpg)


## output
the ROC curve and statistics for each cluster can be found under the `output` directory.
From the labeled images, cluster `0` is white (sky), cluster `1` is gray (ground), and cluster `2` is black (camel).

Additionally, an output log is included to demonstrate what a successful output for this package looks like: `output.log`

![label_example](https://raw.githubusercontent.com/uladkasach/naive-image-segmentation/master/src/images/labels/camel_2.jpg)

 These labeled classes are matched to the segmentation output classes (green, purple, blue) by matching such that accuracy is maximized, as described below.  

## installation and running

#### installation
1. unzip the package
2. ensure python package dependencies are satisfied
    - ensure python version is atleast `2.7.12`
    - ensure packages `Pillow` and `Scikit-Learn` are available
        - e.g., `sudo pip install scikit-learn pillow`

#### running

3. conduct full evaluation
    - `python build_report.py`
4. (opt) run each part individually
    - segmentation
        - note
            - `camel_number` defines which camel to select (valid : 1-10)
            - `blur_number` defines which blur factor to select (valid 0-5)
        - e.g.,
            - `python kmeans.py camel_number blur_number`
            - `python manual_threshold.py camel_number blur_number`
    - evaluation
        - e.g., `python evaluate.py`
            - this automatically evaluates on `kmeans` for `blur_number=0`

**note**: this package caches results. To overwrite cache set the environmental variable `OVERWRITE=true` (e.g., `export OVERWRITE=true`)

## requirements
> You want to evaluate the classification performance of two methods and compare them in terms of accuracy
For the classification problem, consider the segmentation of a grayscale image into THREE classes.
The first method is K-means (K=3), while you may choose any other method as the second classifier; the simplest choice would be setting two intensity thresholds.
Your ground-truth should be the image manually segmented.


## completed deliverables
- generating ground truth data
    - i found 10 images of camels online
        - see `/images/source`
        - camels are typically pictured in a desert with the sky in the background, an ideal choice for segmenting images into 3 classes
            - deserts are typically one color (sandy brown)
            - camels are darker than sand
            - sky is a very distinct different color
    - i manually labeled the images with black, gray, and white
        - see `/images/labels`
- generate 'blurred' sources
    - n order to conduct roc we need to vary something. i will be varying the source images by blurring them as recommended by the professor.
- functional code
    - image preprocessing (`images/loader.py`)
        - this takes care of loading the images, unwraping the images from matrix form into list form, rewraping labeled/modified images from an array and back into matrix form (see `/images/pred`)
    - segmentation (`kmeans.py` and `manual_threshold.py`)
        - segments source images. generates predictions under `/images/pred`
        - running `python kmeans.py 1` will create the image `/images/pred/kmeans/camel_1.jpg`
        - running `python manual_threshold.py 1` will create the image `/images/pred/manual_threshold/camel_1.jpg`
    - evaluation (evaluation.py)
        - utilizes labels and predictions to evaluate TP, TN, FP, and FN
            - TP, TN, FP, FN are calculated for each cluster and each image
                - the stats for each image are normalized w/ respect to the number of pixels before adding to the average amount calculated for the cluster across all images of that blur intensity
                - **note**, we match the label clusters and the predicted clusters by permutating through all of the possible correlations and selecting the permutation that produces the largest accuracy.
    - `build_report.py`
        - this script will have the functionality that will run segmentation on the remaining blurred images not yet segmented, then run evaluation on each set, then combine the results into an ROC curve
