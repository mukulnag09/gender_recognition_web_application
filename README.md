# Gender Recognition using Voice
This repository is about building a deep learning model using TensorFlow 2 to recognize gender of a given speaker's audio. Read this [tutorial](https://www.thepythoncode.com/article/gender-recognition-by-voice-using-tensorflow-in-python) for more information.

## Requirements
- TensorFlow 2.x.x
- Scikit-learn
- Numpy
- Pandas
- PyAudio
- Librosa

Cloning the repository:

    git clone https://github.com/mukulnag09/gender_recognition_web_application/

Installing the required libraries:

    pip3 install -r requirements.txt

## Dataset used

[Mozilla's Common Voice](https://www.kaggle.com/mozillaorg/common-voice) large dataset is used here, and some preprocessing has been performed:
- Filtered out invalid samples.
- Filtered only the samples that are labeled in `genre` field.
- Balanced the dataset so that number of female samples are equal to male.
- Used [Mel Spectrogram](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html) feature extraction technique to get a vector of a fixed length from each voice sample, the [data](data/) folder contain only the features and not the actual mp3 samples (the dataset is too large, about 13GB).

If you wish to download the dataset and extract the features files (.npy files) on your own, [`preparation.py`](preparation.py) is the responsible script for that, once you unzip it, put `preparation.py` in the root directory of the dataset and run it. 

This will take sometime to extract features from the audio files and generate new .csv files.

## Training
You can customize your model in [`utils.py`](utils.py) file under the `create_model()` function and then run:

    python train.py

## Testing

[`test.py`](test.py) is the code responsible for testing your audio files or your voice:

    python test.py --help

**Output:**

    usage: test.py [-h] [-f FILE]

    Gender recognition script, this will load the model you trained, and perform
    inference on a sample you provide (either using your voice or a file)

    optional arguments:
    -h, --help            show this help message and exit
    -f FILE, --file FILE  The path to the file, preferred to be in WAV format

- For instance, to get gender of the file `test-samples/27-124992-0002.wav`, you can:

      python test.py --file "test-samples/27-124992-0002.wav"

    **Output:**

      Result: male
      Probabilities:     Male: 96.36%    Female: 3.64%
  
  There are some audio samples in [test-samples](test-samples) folder for you to test with, it is grabbed from [LibriSpeech dataset](http://www.openslr.org/12).
- To make inference on your voice instead, you need to:
      
      python test.py

    Wait until you see `"Please speak"` prompt and start talking, it will stop recording as long as you stop talking.

## Objective :
To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.

## About the Project :
In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.

## Dataset :
For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it here. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.

## Additional Python Libraries Required :
OpenCV
   - pip install opencv-python
argparse
   - pip install argparse
## The contents of this Project :
- opencv_face_detector.pbtxt
- opencv_face_detector_uint8.pb
- age_deploy.prototxt
- age_net.caffemodel
- gender_deploy.prototxt
- gender_net.caffemodel
- detect.py
For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

## Usage :
Download my Repository
Open your Command Prompt or Terminal and change directory to the folder where all the files are present.
Detecting Gender and Age of face in Image Use Command :
  python detect.py --image <image_name>
Note: The Image should be present in same folder where all the files are present

Detecting Gender and Age of face through webcam Use Command :
  python detect.py
Press Ctrl + C to stop the program execution.
