# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/cropping.JPG "Grayscaling"
[image3]: ./examples/model_loss.png "Recovery Image"





---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The comments also contains some of the problems I meet while coding and how I manage to solve them as a remainder, in case I may meet them again.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of five convolution layers with 3x3 and 5x5 filter sizes and depths of 24, 36, 48 and 64 (model.py lines 101-123)
Then it comes with the five fully connected layers.

The model includes RELU layers(I also tried elu layers as an activation method, but the result seem to be no different) to introduce nonlinearity (code line 101-117), and the data is normalized in the model using a Keras lambda layer (code line 105). Then the picture is cropped using a Cropping2D layer just after the lambda layer, then it comes to the convolutional layer (code line 106).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 113). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 41). The training and validation set are got by first randomly shuffle the original data and then pick 80% of them as an training set and 20% of them as the validation set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I just used the sample_training_data given, the result look fine.
Some techniques about data collection includes driving both conterclockwise and clockwise(that's two tracks of data). Collecting data of driving from the edge of the road to the middle at some sharp turns.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to using preprocessing to generate more data, using pictures take by all three cameras. Using a good model and then tune it.

My first step was to use a convolution neural network model published by NVIDIA in the paper "End to End learning for Self-Driving Cars" I thought this model might be appropriate because the model is designed as an end to end driving solution which also takes pictures from cameras an the input.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by a rate of 0.8. I found that my first model had a low mean squared error on the training set and also a low mean squared error on the validation set. (Of course I applied dropout layers)

Then I used a generator to feed data to the neural network. Using the appropriate parameters to train the model. Tarin the model for 3 epochs with an batch size of 128. the number of training and validation steps per epoch are both 140.

The final step was to run the simulator to see how well the car was driving around track one. It doesn't preform well because I do missed one fully connected layer at first. After I added the layer, the result seem good.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 101-123) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer             		|     Description	        	            				| 
|:---------------------:|:---------------------------------------------:| 
| lambda_1		        	| normalization  					                			|
| cropping2d_1        	| inputs 160x320x3, outputs 89x320x3       			|
| Input         		    | 89x320x3 RGB image   			    		        		| 
| Convolution 5x5     	| 2x2 stride, same padding, outputs 45x160x24	  |
| RELU				        	| activation    					                			|
| Convolution	5x5      	| 2x2 stride,  outputs 23x80x36         				|
| RELU				        	| activation    					                			|
| Convolution 5x5	      | 2x2 stride,  outputs 12x40x48 		        		|
| RELU          	      | acitvation                     		        		|
| Convolution 3x3       | 1x1 stride,  outputs 12x40x64  		        		|
| RELU          	      | acitvation                     		        		|
| Convolution 3x3       | 1x1 stride,  outputs 12x40x64  		        		|
| RELU          	      | acitvation                     		        		|
| Dropout			        	| keep_probability = 0.5                  			|
| Flatten       	      | input 12x40x64, outputs 30720  	         			|
| Fully connedted	      | input 30720,    outputs 1164   		        		|
| Fully connected		    | input 1164,     outputs 100 		          		|
| Fully connected		    | input 100,      outputs 50  		          		|
| Fully connected		    | input 50,       outputs 10  		          		|
| Fully connected		    | input 10,       outputs 1   		          		|
Total parameters: 36,012,663


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.
But I found it really difficult to control the car and I simply do not want to buy a joystick just for that....
So I just use the sample data given as a resources by udacity. And the data is good!

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would make the model more robust and accurate.
I put the code in an function called preprocess_image (code line 81-92).
Cropping is also a kind of preprocessing method. But it is included in the keras layers.
here is an example of the picture after cropping:

![alt text][image2]

But I tried both with and without preprocessing, the car seem to perform the same.....

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by after the third epoch, the loss remains stable and will not go down anymore as I tried. I used an adam optimizer so that manually training the learning rate wasn't necessary.
Shown below is the loss per epoch:

![alt text][image3]
