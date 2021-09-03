# Dog Breed Detector

## Table of Contents

1. [Poject](#project)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors](#licensing)
6. [Repository](#repository)

## Project <a name="project"></a>

This project is part of the Udacity Nanodegree Data Scientist course.
The build model anlyses images. When a dog or a human face is detected the resembling dog breed is returned.

## Instructions <a name="instructions"></a>

The code runs with Python versions 3.9 in a conda environment.<br>
Libraries used in this project are given in dogvenv.yml

1. To create a conda virtual environment and install all required packages run the folowing command in choosen directory. [Additional information](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file):<br>
	`conda env create -f dogvenv.yml`

2. The provided flask app can be started in the app's directory via:<br>
    `python application.py`

3. Go to localhost to view the webpage:<br>
    http://0.0.0.0:3001/

4. Upload a dog image on the webpage UI for dog breed classification.

## Files <a name="files"></a>

The application.py file initiates the package structure and loads the run.py file. The run.py file loads the function for dog, human face and finally dog breed classification. These functions called based on user image input on the webpage, which is finally started.


HTML file:<br>
The index.html webpage shows a dog image and according breed. Below that the UI for the image upload is placed. By uploading an image the routed /go function is triggered to run the classification in run.py. Based on returns the dog breed, the human faces resampling dog breed or a error / try again message is prompted together with the uploaded image. 


Start app via application.py:
Initiates the package structure, the flask class and starts the control file run.py.
Finally the host for the webpage (currently localhost) is started. 

Control flow of backend functions and interface to frontend via run.py:
First the flask route for the index.html webpage is set. Following that the /go function route is defined. It is called after image upload. Then it checks if a file exists, calls the dog and human detector modules before determinig the dog breed via predict_dog.py. In case no dog or human face is detected it returns None otherwise the according dog breed.
Finally the routed function uploaded file sets the path to the uploaded image for use in the html file.

Detect a dog via dog_detector.py
The ResNet50 with the trained 'imagenet' weights is loaded. The uploaded image is passed through the ResNet50 cnn and if index that belongs to a dog class returned the function returns 'True'.

Detect a human faces via human_detector.py:
The pre-trained OpenCV frontal face cascade classifier is loaded and used to detect human faces on the uploaded image. It returns 'True' when at least one human face is detected in image.
The pre-trained OpenCV classifier is stored in the file haarcascade_frontalface_alt.xml.

Determine dog breed via predict_dog.py:
Passes the uploaded image through extract_bottleneck_features.py to get the feature extraction tensor from ResNet50 base model. This tensor is passed for classification through a dog breed classfication cnn to determine and return the dog breed.
The classification cnn was trained on top of ResNet50 base model for classification of dog breeds only.

Base model for feature extraction via extract_bottleneck_features.py:
Module that loads the imagenet pre-trained ResNet50 base model (convolutional part only) for feature extraction, preprocesses the supplied image for usage with ResNet50 base model and returns the tensor passed through ResNet50 base model.

## Results <a name="results"></a>

The ResNet50 based dog detector (dog_detector.py) detected ~99% dogs in the test set correctly and misclassiefied only ~1% of the human faces as dogs. The human face detector (human_detector.py) detected ~99% of all human test images but also classiefied ~12% of the dogs as humans.<br>
The dog detector shows a lower misclassiefication rate and hence was used as main parameter if a message returns that a dog or a human was detected. Based on that the output message with the dogs dog breed or the human resampling dog breed is shown.
The dog breed prediction cnn classifier (predict_dog.py) achieved a prediction rate of ~83%.

Around 8000 dog images where used to train, validate and test the exchanged ResNet50 top layer for classification. The prediction rate might be improved when image augmentation is used. Therefore a change from ResNet50 base model feature extraction to real-time image processing with ResNet50 base model + adapted top layer would be required. This was not done yet.

## Licensing, Authors <a name="licensing"></a>

Used data were provided within the Udacity, Inc. Nanodegree Data Scientist course with a modules project task.

## Repository <a name="repository"></a>

All files required to run the program as per [Instructions](#instructions) are stored in the github repository:<br>
https://github.com/mhoenick/drpp.git
