## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map_output_bboxes.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the Jupyter notebook, `get_hog_features()` function was defined to retunre HOG deatures from an image.  

In cell 6, I started by reading in all the `vehicle` and `non-vehicle` images. The are 8792 vehicle and 8968 non-vehicle images. Most of the vehicle images were taken from the rear side.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the one we used in the class exercise seemed to work fine. The orientation is usually within (9, 12), but the since orient=9 already showed fairly good result, I did not increase further due to efficient consideration.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier training code is in the 8th cell of Jupyter notebook. I trained a linear SVM using the paramters showed in below table and combined all the features into one-colume feature vector.

First, I read in all images from vehicle and non-vehicle class, then extracted features from each of them. Later, I split the datasets to 80%/20% for training and test with random shuffle by using `sklearn.model_selection.train_test_split()`.

I also used the `sklearn.preprocessing.StandardScaler()` function to normalize the data from training set. The obtained scaler would be applied for test data and later for all the frame we feed in the process pipeline.

The fianl `Feature vector` was with 8460 of length and the test accuracy of my SVC classifier is close to 98.7%.

| Parameter        | Value           |
| ---------------- |:---------------:|
| color_space      | 'YCrCb'         |
| orient           | 9               |
| pix_per_cell     | 8               |
| cell_per_block   | 2               |
| hog_channel      | 'ALL'           |
| spatial_size     | (32, 32)        |
| hist_bins        | 32              |
| spatial_feat     | True            |
| hist_feat        | True            |
| hog_feat         | True            |

To save time, I saved trained result as a pickle file and retrieved all the characteristics every time I run the code.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for my sliding window search is in the 9th-11th cells. The implementation was largely referred to class examples and exercises.

First, I limited the search area to be within (400, 656) pixel in Y since we want to avoid searching on the sky and tree tops. Then I tried various scales and calcualted the number of windows need to be searched.

After couple iterations, I decided to use 2 scales of 1.5 and 2.0 with different ystart/ystop listed in below table. Since vehicle that closer to the bottom of image is likely to have bigger scale, I cut the 1.5 scale search at y=592 to improve efficiency.

| Scale        | ystart           | ystop           | No. of Windows  |
| -------------|:----------------:|:---------------:|:---------------:|
| 1.5          | 400              | 592             | 250             |
| 2.0          | 400              | 656             | 185             |

The total windows for a full window search frame ultimately was 250+185=435. A visulization of my sliding window search is showed in the below image.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

I do see some false positives here and there especially under the shadow, but I was not too worried about them since they could be easily filtered out after integrated heatmaps over several frames.

Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The codes of filter for false are located in the 16th-22th cells for Jupyter notebook.

To keep better tracks of detected cars and integrated heatmap, I defined 2 classes called `Vehicle()` and `Heatmap()`.

I recorded the positions of positive detections in each frame of the video. I then created an integrated heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The codes of my pipeline and some helper functions/classes are located in the 19th-22th cells for Jupyter notebook.

To keep better tracks of detected cars and integrated heatmap, I defined 2 classes called `Vehicle()` and `Heatmap()`.

In my pipeline, for positive detections in each frame, I used them to update the `Heatmap()` object called `integrated_heatmap` which would keep tracks of heatmaps over the last 6 frames. Inside the `Heatmap().update()` function, I did the same work as before to filter out false positives. Then used them to construct a list of high confidence detections.

Once I have a list that containing bboxes detected in the current frame, then I tried to match them with the previously detected ones. If I found a match (the ones with minimum distance from last frame), I updated that car with the new information obtained. If not, I considered a new car found and then created new `Vehicle()` object and append to `detected_carslist`. Ultimately, all detected vehicles would be stored in the `detected_carslist`.

After updating with the latest detections, I looped through all detected vehicles and did the following:

1. If no match found from current frame: record number of consecutive missed frame.

2. If a vehicle missed <= 5 frames, still keep it alive. If more than 5, remove this car from `detected_carslist`.

3. For all vehicles that I have high confidnence they are in the current frame, I record their bounding boxes and draw on the original image frame.

To better improve the efficiency, I also added two workarounds:

1. Before each frame, I first checked if there is any record in the `detected_carslist`, if no, I only do the search every 10 frames. This helps a lot when there is no any other cars in the video for a long time.

2. I also only do full window search on the first frame and every 10 frames after. For all other frames in between, I only search the nearby area of previously detected vehicles. This also much reduced the number of windows need to be searched.

After all this, I got the pipeline working nicely. The detection was clean, smooth and accurate for most of the time.

What I think could be the further improvements are:

1. The detections were poor when other vehicles showing sides to the camera view: 

This is mainly due to the training data. Most of them are showing the rear side of the vehicle. To better generalize my classifier, we need more data to train and also data from other sides of view.

2. The pipeline could become slow when there are many vehicles, this can also degrade the accuracy: 

I probably can make use of threshold of number of sliding windows, once exceeded some limit, just turn back to full search (435 windows). We also need a way to deal with vehicles that are overlapped.
