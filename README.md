# Dog Breed Detector

## Table of Contents

1. [Poject Overview](#project)
	- [Problem Statement](#problem)
	- [Metrics](#metrics)
	- [Data Exploration and Visualization](#explore)
	- [Data Preprocessing](#preprocessing)
	- [Implementation](#implementation)
	- [Refinement](#refinement)
	- [Model Evaluation and Validation](#model)
	- [Justification](#justification)
	- [Reflection](#reflection)
	- [Improvement](#improvement)
2. [Instructions](#instructions)
3. [Licensing, Authors](#licensing)
4. [Repository](#repository)

## Project Overview <a name="project"></a>

This project is part of the Udacity Nanodegree Data Scientist course.
An image classification model and a web app as user interface is build. When a dog or a human face is detected in the uploaded image the resembling dog breed is returned.

The dog app detection model creation was done within the jupyter notebook 'dog_app.ipynb', which was provided by Udacity. It includes details of the steps taken from data preprocessing to model generation and finally testing. The used datasets are not uploaded to github.


### Problem Statement <a name="problem"></a>

The classifier should detect dogs or human faces and output the resampling dog breed. Therefore three different algortithms will be used. First the pictures will be scanned if a dog or human face can be detected at all. To do that a CNN for dog detection and a classifier for human face detection shall be used. Both will return True or False respectively. 
If neither was detected an error message will be printed. Otherwise, the dogs dog breed or resampling humans dog breed should be predicted. For that a CNN should be trained to distinguish between 133 dog breeds and return the according one.

For human face detection the effective Haar feature-based cascade classifier available via OpenCV will be used.
For dog detection the at keras available ResNet50 CNN, trained on ImageNet, will be used. For dog detection only this is possible as we only need to check if the identified class index is within the dogs classes of ImageNet. As per [ImageNet](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) classes definition those are from 151 to 268. 

Finally the base layer of the ResNet50 CNN, trained on ImageNet, will be used for transfer learning via feature extraction. All training, validation and test images passed through the base model once and output tensors were saved. 
A new top layer is build, that reads these tensors and reduces them via dense layers to the required number of classes.


### Metrics <a name="metrics"></a>

For dog and human face detection the accuracy will be measured. This is sufficient as for testing either only dog images or only human images will be presented. Hence, all or non should be classified correct or false. 

Accuracy = correctly classified's / total images * 100


For the dog breed detection model also the precision and recall will be measured during model training and testing. 
As shown below in [Data Exploration and Visualization](#explore) the dog breed image data set is a bit imbalanced. Accuracy 

The distribution of the training dog breed images is shown below in [Data Exploration and Visualization](#explore). The set is slightly imbalanced.
Accuracy considers only total correctly classified over total images and hence is not representative for accuracy of imbalanced data. For imbalanced data it is required to evaluate the accuracy per class and then average. Therefore precision and recall are added for dog breed classification model training. Precision and recall consider correctly classified images per class before averaging.

Precision = Sum of (true positives per class) / 
			(Sum of (true positives per class) + Sum of (false positives per class))

Recall = Sum of (true positives per class) /
		( Sum of (true positives per class) + Sum of (false negatives per class))

### Data Exploration and Visualization<a name="explore"></a>

The dog dataset consist of 8351 dog images with 133 different dog breed categories. It is split into train, validation and test set. Whereby the sets made of 80%, 10% and 10%, respectively.

The dog images are stored in folders named after the stored dog breed. The folder names will be extracted and used as resulting breed for model training and testing.
Folder structure is setup like:<br>
Dog Images<br>
&emsp;&emsp;|--- Train Set<br>
&emsp;&emsp;&emsp;&emsp;|--- Dog Breed Name 1<br>
&emsp;&emsp;&emsp;&emsp;|--- Dog Breed Name 2<br>
&emsp;&emsp;&emsp;&emsp;|...<br>
&emsp;&emsp;|--- Test Set<br>
&emsp;&emsp;&emsp;&emsp;|...<br>
&emsp;&emsp;|...<br>


For the test set distribution of the number of dog images per dog breed are shown in the chart below. It shows, that the training set for dog breed classification is imbalanced. At minimum 26 images per breed against a max of 77 for some dog breeds are available. The mean with 50.2 and the median with 50 are pretty equal. Therefore the dataset is not skewed but due to the difference of 51 between min and max images per dog breed the data is imbalanced. 

To see if this imbalance already affects the dog breed classification model, precision and recall are returned next to accuracy during model training. If they are poor, the train data has to be adapted.

<p align="center">
<b>Chart 1: Train Set: Available Images per Dog Breed</b><br>
<img src="dogapp/static/DogImages_distrib2.png" width="400">
</p>


The human face dataset includes 13233 images in total, but only a few will be used for testing accuracy. A result vector is not needed as for testing only images including human faces will be presented.


### Data Preprocessing<a name="preprocessing"></a>
Dog detector via ResNet50:
The 'imagenet' trained ResNet50 model loaded from keras requires a 4D tensor (number of images, image height, image width, number of channels) as input. In dog_detector.py the image is loaded, resized to a 224 x 224 image and converted to the 4D tensor by the path_to_tensor() function.
Additionally the keras ResNet50 input needs to be converted from RGB to BGR and each color channel is zero-center with respect to the ImageNet dataset without scaling. This will be done by the keras resnet50.preprocess_input() function. 

Human face detector via Haar cascade classifier: 
The already trained haar cascades classifier that is used requires only a conversion from color to grayscale images.

Dog breed prediction: 
As ResNet50 base model was used here upfront of the adapted top layer CNN, the same input preprocessing as for the "dog dector via ResNet50" above was done.
For the training of the adapted top layer the previously extracted feature tensors of the images were used. As they are originally the output of the ResNet50 base model they could be used directly.


### Implementation<a name="implementation"></a>

Start app via application.py:<br>
Initiates the python package structure, the flask class and starts the control file run.py. Finally the host for the webpage (currently localhost) is started.

Control flow of backend functions and interface to frontend via run.py:<br>
First the flask route for the index.html webpage is set. Following that the /go function route is defined. It is called after image upload. Then it checks if a file exists, calls the dog and human face detector modules before determinig the dog breed via predict_dog.py. In case no dog or human face is detected it returns none otherwise the according dog breed.
Finally the routed function uploaded file sets the path to the uploaded image for use in the html file.

Detect a dog via dog_detector.py<br>
The ResNet50 with the trained 'imagenet' weights is loaded. The uploaded image is passed through the ResNet50 cnn and if it returns an index, that belongs to a dog class of the ImageNet categories, the function returns 'True'.

Detect a human faces via human_detector.py:<br>
The trained OpenCV frontal face cascade classifier is loaded and used to detect human faces on the uploaded image. It returns 'True' when at least one human face is detected in image.
The trained OpenCV classifier is stored in the file haarcascade_frontalface_alt.xml.

Determine dog breed via predict_dog.py:<br>
Passes the uploaded image through extract_bottleneck_features.py to get the feature extraction tensor from ResNet50 base model. This tensor is passed for classification through a dog breed classfication cnn to determine and return the dog breed.
The classification cnn was trained on top of ResNet50 base model for classification of dog breeds only.

Base model for feature extraction via extract_bottleneck_features.py:<br>
Module that loads the imagenet pre-trained ResNet50 base model (convolutional part only) for feature extraction, preprocesses the supplied image for usage with ResNet50 base model and returns the tensor passed through ResNet50 base model.

HTML file for UI:<br>
The index.html webpage shows a dog image and according breed. Below that the UI for the image upload is placed. By uploading an image the routed /go function is triggered to run the classification in run.py. Based on returns the dog breed, the human faces resampling dog breed or an error / try again message is prompted together with the uploaded image.
If the upload / reset button is triggered without a specified image the /go function routes back to start page and cleans the uploads image folder internally.

Dog, human detector and prediction function development within dog_app.ipynb/html:<br>
Within the jupyter notebook file the dog and human face detector and dog prediction models with the required preprocessing are developed and tested. It is supplied for reference only. The required data folders to rerun it are not uploaded to github.
The dog_app.html file stores the achieved outputs for reference.


### Refinement<a name="refinement"></a>

For dog breed classification three models were tested. A self build CNN related to VGG architecture, a VGG16 and a ResNet50 model with an adapted top layer for dog breed classification. For comparison all were trained for 20 epochs.

The self build CNN details can be reviewed in the dog_app.html file. In summary it uses three repetitions of two convolutional layer and a max pooling layer, and ends with two dense layers to reduce to the number of required classes.

For comparability the VGG16 and ResNet50 use the same top layer architecture. It consist of a global average pooling and a dense layer with the 133 output classes.

The table below shows the results of five runs for each of the compared models. Yellow highlights the max values achieved from five runs per model. Green highlights the max average value of all models. The ResNet50 model achieved the best dog breed classification and is robust, hence I moved on with it.

<p align="center">
<b>Table 1: Model Performances</b><br>
<img src="dogapp/static/ModelEval1.png" height="400">
</p>


The ResNet50 transfer learning based model training accuracy with up to 99% indicated overfitting. Hence I tried to adapt the top layer for more generalization.

The different adaption approaches results are shown in the table below. The first model is the ResNet50 with the global average pooling and dense (133) as top layer from above.
First I exchanged the top layer with a set of dense layers, each followed by a dropout layer but it decreased prediction accuracy.
Next I used a global max pooling instead of global average pooling, which resulted in a similar performance to the first model but still lower.
Next I combined the global average pooling layer with a dropout layer with different dropout probabilities. That showed no performance improvement as well, hence the first model with a prediction accuracy of ~83% was kept.

<p align="center">
<b>Table 2: ResNet50 Top Layer Refinement - Performances</b><br>
<img src="dogapp/static/ModelEval2.png" height="700">
</p>


### Model Evaluation and Validation<a name="model"></a>

Human face detector: 
With a detection rate of ~99% for human faces the OpenCV haar cascade classifier approach is a good option to use. However, due to the misclassification of ~12% of the dog images as human faces an upfront elimination of all dog images with the dog detection algortithm is required for the web app.

Dog detector: 
The dog detection based on ResNet50 model worked really good. From 100 test images it detected 100% with a misclassification rate of 0%. For more images it still achieved ~99% with a misclassification of only ~1%. Hence it is used for detecting all dogs before the image is scanned for human faces and should therefore avoid misclassification of dogs as humans.

Dog breed prediction:
The final model selected for dog breed prediction is the ResNet50 base model plus a global average pooling layer and a dense layer with the according 133 output classes. Trial 4 of it (first model) in table 2 is used. 
As shown in table 2 above the choosen model achieved the best average accuracy, -precision and -recall on the test set of all tested model variations.
On the test and validation set the average precision and recall are all over 80% and close to the accuracy. For the equally weighted averaged precision and recall the data imbalance could even out, hence for the finally choosen model the weighted precision with 86.5% and the weighted recall with 84.5% were calculated to cover imbalance. These values are in line with the overall performance of the model hence the lower number of images for some dog breeds are still enough to achieve reasonable predictions.

From the five runs performed and shown in table 2 for the first model it can be seen that the model is robust as all five run performances are close to each other. 


### Justification<a name="justification"></a>

The dog and human face detection models work very good with a ~99% detection rate and in combination they should allow distinguishing between dog and human faces with the low misclassification rate of the dog detector with ~1%.
For dog breed classification the model with the best accuracy out of the tested once is used. With ~83% it shows a reasonable classification accuracy.
Compared to the [ResNet50](https://arxiv.org/abs/1512.03385) top 1 and top 5 error rates of 22.8% and 6.7% respectively an error rate of ~17% is quite good.

### Reflection<a name="reflection"></a>

It was a very interesting challenge. It forced me to read more papers about known CNN architectures like VGG to get a better understanding why architectures are build as they are. Building a CNN from that quickly reached the limit of available GPU ressources. But allowed me already to see how difficult it is to find the right setup of convolutional, dropout and pooling layers to get a reasonable result and not run into overfitting.
Seeing than the huge improvement that can be gained quite easily via transfer learning from a pre-trained CNN architecture like ResNet50 or VGG compared to the results from a self build CNN with the same gpu resources is impressive.
That showed quite good that for solving challenging computer vision tasks the knowledge about good CNN architectures is a key requirement.


### Improvement<a name="improvement"></a>

The dog breed prediction is quite good but fails on some pictures or do not predict the sub dog breeds correctly. The difference in accuracy between trainig and validation set indicates overfitting. Image augmentation should help to better generalize.
Therefore a change from ResNet50 base model feature extraction to real-time image processing with ResNet50 base model + adapted top layer would be required. This was not done yet, but is planned.


## Instructions <a name="instructions"></a>

The code runs with Python versions 3.9 in a conda environment.<br>
Libraries used in this project are given in dogvenv.yml

1. To create a conda virtual environment and install all required packages run the folowing command in choosen directory:<br>
	`conda env create -f dogvenv.yml`

2. The provided flask app can be started in the app's directory via:<br>
    `python application.py`

3. Go to localhost to view the webpage:<br>
    http://0.0.0.0:3001/

4. Upload a dog image on the webpage UI for dog breed classification.


## Licensing, Authors <a name="licensing"></a>

Used data were provided within the Udacity, Inc. Nanodegree Data Scientist course with a modules project task.


## Repository <a name="repository"></a>

All files required to run the program as per [Instructions](#instructions) are stored in the github repository:<br>
https://github.com/mhoenick/dogapp