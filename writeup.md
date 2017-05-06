
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/original_car_img_x.png
[image11]: ./output_images/original_notcar_img_x.png
[image2]: ./output_images/car_notcar_hog_plot_x.png
[image3]: ./output_images/test5.jpg
[image4]: ./output_images/test6.jpg
[image5]: ./output_images/test_image_0_heat_map.png
[image51]: ./output_images/test_image_heat_map.png
[image6]: ./examples/labels_map.png
[image7]: ./output_images/final_pipeline_image_39.png
[image71]: ./output_images/final_pipeline_plot_24.png
[image72]: ./output_images/final_pipeline_plot_27.png
[image73]: ./output_images/final_pipeline_plot_30.png
[image74]: ./output_images/final_pipeline_plot_33.png
[image75]: ./output_images/final_pipeline_plot_36.png
[image76]: ./output_images/final_pipeline_plot_39.png

[video1]: ./processed_project_video.mp4

#### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---
### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the first code cell of the IPython notebook.
Please refer to the function with the following interface:

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,
                    feature_vec=True):
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

vehicle

![car][image1]

non-vehicle

![not car][image11]

Below are some data characteristics used for training the classifier. One important remark is that the data for vehicle and non-vehicle is balanced, number of features around 9 thousands for each class. As explained in the lesson a balanced data set is important for training of a classifier.

| Characteristic        | value           |
| ------------- |:-------------:|
| non-vehicle      | 8968 |
| vehicle      | 8792|
| data_type | float32|
|image_shape|(64, 64, 3)|


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

| Characteristic        | value           |
| ------------- |:-------------:|
| colorspace      | YCrCb |
| orientations      | 9|
| pix_per_cell | (8,8)|
|cell_per_block|(2,2)|
|hog_channel|ALL|



#### 3. Describe how (and identify where in your code) you trained a classifier

I trained a linear SVM using HOG features combined with spatial features and histogram features.
Please refer in first code cell in the notebook the following function for extracting the features:

```python
def extract_features(imgs,
                     cspace='BGR2YCrCb',
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel = 0,
                     spatial_size=(32, 32),
                     hist_bins=32):
```
In the fifth notebook code cell you find a function with the following interface:

```python
def train_svc(cars, notcars):
```
In this function the classifier is trained and tested. The resulting trained classifier and the scaler are store to a pickle for later use.

A StandardScaler is used to normalize feature data.

```python
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
```
Split up data into randomized training and test sets.
```python
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                    test_size=0.2,
                                                    random_state=rand_state)
```

Feature vector length: 8460

26.5 Seconds to train SVC...

Test Accuracy of SVC =  0.9916

My SVC predicts :  [ 0.  1.  1.  0.  1.  0.  1.  1.  0.  0.]

For these 10 lab:  [ 0.  1.  1.  0.  1.  0.  1.  1.  0.  0.]

The trained classifier returns a very good test accuracy of almost 100%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this:

| Characteristic        | value           |
| ------------- |:-------------:|
| ystart      | 400 |
| ystop      | 656|
| scale | 1.5|

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

![alt text][image51]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

![alt text][image71]

![alt text][image72]

![alt text][image73]

![alt text][image74]

![alt text][image75]

![alt text][image76]
### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
