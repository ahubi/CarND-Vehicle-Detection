
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
[image3]: ./output_images/test4.jpg
[image4]: ./output_images/car_detection_test_imagestest1.jpg
[image41]: ./output_images/car_detection_test_imagestest3.jpg
[image42]: ./output_images/car_detection_test_imagestest5.jpg

[image5]: ./output_images/test_image_0_heat_map.png
[image51]: ./output_images/test_image_heat_map.png
[image7]: ./output_images/final_pipeline_image_38.png
[image71]: ./output_images/final_pipeline_plot_26.png
[image72]: ./output_images/final_pipeline_plot_28.png
[image73]: ./output_images/final_pipeline_plot_30.png
[image74]: ./output_images/final_pipeline_plot_32.png
[image75]: ./output_images/final_pipeline_plot_34.png
[image76]: ./output_images/final_pipeline_plot_36.png

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
In this function the classifier is trained and tested. The resulting trained classifier and the scaler are stored to a pickle for later use.

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

Below is the of a trained classifier

Feature vector length: 8460

26.5 Seconds to train SVC...

Test Accuracy of SVC =  0.9916

My SVC predicts :  [ 0.  1.  1.  0.  1.  0.  1.  1.  0.  0.]

For these 10 labels:  [ 0.  1.  1.  0.  1.  0.  1.  1.  0.  0.]

The trained classifier returns a very good test accuracy of almost 100%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  
I decided to search lower window positions since here car will appear in the images taken by the car camera. Searching for cars in the sky and in the trees doesn't make sense. In the image below you can see the region with red squares which is scanned and where cars are detected - more highlighted squares. In case of scale factor 1 and cells_per_step of 2 overlap of 75% is the result during window search.

| Characteristic        | value           |
| ------------- |:-------------:|
| ystart      | 400 |
| ystop      | 656|
| scales | 1.5, 1|
| cells_per_step | 2|


![alt text][image3]

Please refer in the notebook to the function with the following interface which contains the implementation of an algorithm to find cars in an image. The function was provided during Udacity class but was slightly modified to return car boxes instead of images.

```python
def find_cars(img, ystart, ystop,
              scale, svc, X_scaler,
              orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins):
```
The function takes the image on which the cars should be searched and parameters required for the algorithm like image region, trained classifier, scaler and parameters how to extract features from the image regions.

Remark: The parameters to extract features must match the parameters which where used during the training of the classifier.   

#### 2. Show some examples of test images to demonstrate how your pipeline is working.
Ultimately I searched on two scales (1, 1.5) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

Below is an image where two car of different car are clearly detected.

![alt text][image4]

In the next image is a car far away from the camaera and also clearly detected.

![alt text][image41]

In the last image you can see two detected cars and additionally two 'false positives' in the shadow of the trees. In the next section an approach is described how to minimize the 'false positives'.

![alt text][image42]
---

### Video Implementation

#### 1. Final video output.  
Here's a [link to my video result](./processed_project_video.mp4)

#### 2. Filter for 'false positives' and combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here is an example of one frame with two detected cars and two 'false positives':

![alt text][image5]

After applying a threshold of 1 to the same frame the image looks different and the false positives are no more visible.

![alt text][image51]

I constructed bounding boxes to cover the area of each blob detected. To combine detected rectangles OpenCv function groupRectangles is used. To track the pipeline I created some parameters.

```python
#threshold
threshold_frames = 1
```

'threshold_frames' us used to control how many frames must be collected before the heat algorithm is applied to the detected boxes on the frames.


```python
threshold_boxes = 3
```
'threshold_boxes' is used to control how many 'labeled boxes' must be collected before the rectangles are grouped together.

To control heat threshold I use this formula.

```python
# Apply threshold to help remove false positives
heat = apply_threshold(heat,threshold_frames*4)
```


The source for tracking the pipeline is found in the last two cells. Please refer to the following functions:

```python
def get_labeled_bboxes(labels):  
def process_image(img):
```

### Here is the output of last 10 frames from test_video.mp4 file processed through my final pipeline:

![alt text][image71]

![alt text][image72]

![alt text][image73]

![alt text][image74]

![alt text][image75]

![alt text][image76]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

During test on the pipeline I noticed that false positives can be rejected by applying higher 'heat' threshold values. However the threshold for heat can't be very hight since then real cars won't be rejected. I think that there might be a better way to extract features and combine them to achieve better predicitons and less 'false positives'.

The way my code combines the boxes on the image has a lot of room to improve or implement a better approach. From time to time more than one box is drawn on the vehicle.
