# Naive Image Segmentation

This project was completed as a homework assignment for the Biometrics course at IUPUI.

This project was developed with Python v2.7.12.

This project is dependent on the python packages `Pillow` and `Scikitlearn`

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
    - segmentation (kmeans.py and manual_threshold.py)
        - segments source images. generates predictions under `/images/pred`
        - running `python kmeans.py 1` will create the image `/images/pred/kmeans/camel_1.jpg`
        - running `python manual_threshold.py 1` will create the image `/images/pred/manual_threshold/camel_1.jpg`
    - evaluation (evaluation.py)
        - utilizes labels and predictions to evaluate TP, TN, FP, and FN
            - TP, TN, FP, FN are calculated for each cluster and each image
                - the stats for each image are normalized w/ respect to the number of pixels before adding to the average amount calculated for the cluster across all images of that blur intensity
    - image preprocessing (images/loader.py)
        - this takes care of loading the images, unwraping the images from matrix form into list form, rewraping labeled/modified images from an array and back into matrix form (see `/images/pred`)

## left to complete
- `build_report.py`
    - this script will have the functionality that will run segmentation on the remaining blurred images not yet segmented, then run evaluation on each set, then combine the results into an ROC curve
