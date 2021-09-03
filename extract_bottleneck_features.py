def extract_Resnet50(tensor):
    '''
    Image tensor taken and preprocessed to be used with keras ResNet50 Model
    
    Input: np.array (1, 224, 224, 3), image tensor
    Output: np.array (1, 2048), tensor passed trough ResNet50 base model
    '''
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet', include_top=False, pooling='max').predict(preprocess_input(tensor))