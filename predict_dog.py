import numpy as np
from glob import glob
from keras.models import Sequential, load_model
from keras.layers import InputLayer, GlobalMaxPool2D, Dense

# from dog_detector import dog_detector, 
from dog_detector import path_to_tensor
# from human_detector import face_detector
from extract_bottleneck_features import extract_Resnet50


# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]

# Load trained transfer learning top of ResNet50 model for dog classification
Resnet50_model = load_model('saved_models/resnet50')


def Resnet50_predict_breed(img_path):
    '''
    Functions takes a path to an image as input
    and returns the dog breed that is predicted by the model.
    '''
    # extract bottleneck features
    tensor = path_to_tensor(img_path)
    # extract_Resnet50 represents ->
    # ResNet50(weights='imagenet', include_top=False, pooling='max').predict(preprocess_input(tensor))
    bottleneck_feature = extract_Resnet50(tensor)
    #adapt dimensions from (1, 2048) to req'd input shape (1,1,1,2048)
    bottleneck_feature = np.expand_dims(np.expand_dims(bottleneck_feature, axis=0), axis=0)
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split('.')[-1]


# def determine_dog_breed(img_path):
    # """
    # Function to determine whether the image contains a human, a dog, 
    # or neither and returns the dog breed or None
    
    # Input: 
    # img_path = (str), path to image file
    # Output: 
    # dog_breed = (str), detected dog breed
    # """
    # # human = face_detector(img_path)
    # # dog = dog_detector(img_path)
    
    # # if (human == True) or (dog == True):
        # # dog_breed = Resnet50_predict_breed(img_path)
    # # else:
        # # dog_breed = None
    # dog_breed = Resnet50_predict_breed(img_path)
    # return dog_breed