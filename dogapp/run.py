import os
import tensorflow as tf
from flask import render_template, request, redirect, url_for
from glob import glob
from werkzeug.utils import secure_filename

# Set tf gpu memory allocation to growth before tf dependent func's imported
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus[0]))
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Tensorflow GPU Memory Growth Option: enabled")

from dogapp import app
from dog_detector import dog_detector
from human_detector import face_detector
from predict_dog import Resnet50_predict_breed #determine_dog_breed

# folder + extensions for image upload
upload_folder = 'static/uploads/'
allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'tiff'}
app.config['UPLOAD_FOLDER'] = upload_folder


# define allowed file extensions for uploaded file
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_extensions

# initiate index page
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# routine for image upload and dog breed prediction
@app.route('/go', methods=['GET', 'POST'])
def upload_file():
    '''
    Function that is called from frontend via form with action /go 
    Checks if file with allowed extension is provided and 
    detects dog or human face and predicts according dog breed, 
    which is returned with variables to allow printing of different
    statements in dependency of None, dog or human face detected.
    
    Input: img, file uploaded via frontend by user
    Output: 
    - file: werkzeug-Filestorage, filename and type info
    - filename: str, name of uploaded image
    - breed: str, name of detected dog breed
    - human: bool, True if human face detected
    '''
    if request.method == 'POST':

        file = request.files['file']
        upload_path = os.path.join('dogapp', app.config['UPLOAD_FOLDER'])
        
        # if user does not select file, browser also 
        # submit an empty part without filename
        if file.filename == '':
            # remove old files in upload folder
            for img in glob(upload_path + '/*'): 
                os.remove(img)
            # redirect to start / index page
            return redirect('index')
        
        if file and allowed_file(file.filename):
            # check if allowed file extension
            filename = secure_filename(file.filename)

            # save uploaded file
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)
            
            # check if dog or human face in picture
            dog = dog_detector(file_path)
            print(f'\nDog: {dog}\n')
            human = face_detector(file_path)
            print(f'\nHuman: {human}\n')

            # determine dog breed if dog or human face
            #breed = determine_dog_breed(file_path)
            if (human == True) or (dog == True):
                breed = Resnet50_predict_breed((file_path))
            else:
                breed = None
            print(f"\nBreed: {breed}\n")
            
            # print(url_for('.static', filename='uploads/' + filename))
            # print(f'file: {type(file)}, {file}')
            # print(f'filename: {type(filename)}, {filename}')
            # print(f'breed: {type(breed)}, {breed}')
            # print(f'human: {type(human)}, {human}')
            
            # provide variables to frontend index.html
            return render_template("index.html", file=file, filename=filename,  breed=breed, dog=dog ,human=human)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    '''
    Provide path to uploads folder to frontend.
    As required by html, the path is relative to index.html.
    
    Input: str, filename generatd from upload_file func 
    Output: str, relative path to image file
    '''
    return redirect(url_for('.static', filename='uploads/' + filename))