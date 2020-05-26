# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Visualization.JPG "Visualization"
[image2]: ./examples/Grayscale.jpg "Grayscaling"
[image3]: ./examples/Normalized.JPG "Normalized"
[image4]: ./examples/10.png "Traffic Sign 1"
[image5]: ./examples/11.JPG "Traffic Sign 2"
[image6]: ./examples/17.png "Traffic Sign 3"
[image7]: ./examples/3.png "Traffic Sign 4"
[image8]: ./examples/40.JPG "Traffic Sign 5"


---

You're reading it! and here is a link to my [project code](https://github.com/Hyun5/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 1. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are exist in each lable.

![alt text][image1]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because the color does not matter but it reduce the training time.

Here is an example of a traffic sign image before and after grayscaling.

```
### converting to grayscale
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

print('RGB shape:', X_train_rgb.shape)
print('Grayscale shape:', X_train_gry.shape)
```

![alt text][image2]

As a last step, I normalized the image data because it can make sure our data looks and reads the same way across all data sets.

Here is an example of an original image and an augmented image:
```
## Normalize the train and test datasets to (-1, 1)

X_train_normalized = (X_train_gry - 128)/128 
X_test_normalized = (X_test_gry - 128)/128
```

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x6		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  					|
| Flatten		      	|         										|
| Fully connected		| Size: 120										|
| RELU					|												|
| Fully connected		| Size: 84										|
| RELU					|												|
| Fully connected		| Size: 43										|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used 60 epochs, a batch size of 128 and a learning rate of 0.0005.
Based on some trail, I increased the number of epochs and decreased the learning rate from the initial value.

For my training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.

My final model Validation Accuracy was 0.897

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.897 
* test set accuracy of 0.881



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			       	        		        | Prediction									| 
|:---------------------------------------------:|:---------------------------------------------:| 
| No passing for vehicles over 3.5 metric tons	| No passing for vehicles over 3.5 metric tons	| 
| Right-of-way at the next intersection			| Right-of-way at the next intersection			|
| No entry										| No entry										|
| Speed limit (60km/h)					   		| Speed limit (50km/h) 							|
| Roundabout mandatory							| Yield 										|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The top five soft max probabilities were;

```
10.png:
No passing for vehicles over 3.5 metric tons: 100.00%
Dangerous curve to the right: 0.00%
Roundabout mandatory: 0.00%
End of no passing by vehicles over 3.5 metric tons: 0.00%
Slippery road: 0.00%

11.JPG:
Right-of-way at the next intersection: 100.00%
Beware of ice/snow: 0.00%
Dangerous curve to the right: 0.00%
Children crossing: 0.00%
Slippery road: 0.00%

17.png:
No entry: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

3.png:
Speed limit (50km/h): 99.88%
Speed limit (30km/h): 0.12%
Speed limit (60km/h): 0.00%
Yield: 0.00%
Speed limit (80km/h): 0.00%

40.JPG:
Roundabout mandatory: 99.94%
Right-of-way at the next intersection: 0.06%
Speed limit (100km/h): 0.00%
End of no passing: 0.00%
Yield: 0.00%

```


#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From the search, the accuray will be better if I use more preprocessing of dataset, such as transaltion, scaling, warp, etc.
I did not use the dropout. This will also increase the accuracy more by avoiding the overfitting.

