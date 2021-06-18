[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) 

# Traffic_Sign_Classifier-with-Streamlit

## App:
- Link: https://traffic-sign-classification.herokuapp.com/



## App Output
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/Front%20Page.png)
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/Image%20upload.png)
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/Result.png)


## Project overview
- The aim of this project is to focus on the first fundamental features of the decision making ability of an autonomous vehicle, 
i.e., to develop a deep learning model that reads traffic signs and classifies them correctly using Convolutional Neural Networks(CNNs).

- The traffic sign classifier uses a German traffic dataset. The German traffic dataset consists of
34,799 32*32 pixels colored images that is used for the training dataset, 12,630 images are used
for the testing dataset and 4410 images are used in the validation dataset where each images is a
photo of a traffic sign belonging to one of the 43 classes i.e., traffic sign types.


## Dataset:
Dataset can be found here : https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign


## Folders Description
### Google Colab Notebook
#### Contains the whole process of building the CNN Model 
- Load the Pickled dataset
- Use Seaborn to visualise the data.
- Preprocess the images using OpenCV.
- Use ImageDataGenerator for image augmentation and help the model generalise it's results.
- build_model() function takes hyperparameter(hp) as input and we start building our CNN model using KerasTuner and then compile our model.
- KerasTuner gives us the best hyperparameter combinations using RandomSearch method.
- We now create a model checkpoint and then fit the model and run it for 40 epochs.
- Now Load the model's weights and biases and evaluate it on our test dataset.
- Save our model in Keras HDF5 format.
- Use the saved model to test on random images.

### Test Random Images
- This contains images from the internet. A total of 43 images belonging to each class.
- Our model will be tested using this unseen data

### Streamlit App Output
- Contains the App's final output 

### Class Names and Labels
- Contains the signnames.csv file

### Result Excel
- Conatins a exccel sheet having the results of our test results on random images from the internet
- Also contains the accuracy of our model on unseen data
- Accuracy on unseen data : 79.06%
### Model
Contains the saved keras model named
- ###### Traffic_sign_classifier_CNN.hdf5

## Files for our Streamlit App


#### app.py
- Contains the front-end code for the streamlit app.
- Imports the predict() function fetches the result and displays it.
- get_model(): Loads the saved model into cache using streamlit's "@st.cache" feature.
- predict(): Takes an image as input from the function parameter, preprocesses it and feeds it to the model for results.

#### Procfile
A Procfile is a file which describes how to run your application.


## Run this app on your system.
### Requirements
- Python 3.6+
- NumPy
- Pillow
- TensorFlow 2.x
- Streamlit 

### To run it on your system
- Install all the dependencies
- Clone this repository
- You need the Streamlit App folder to run this application.
- In your Command line/Terminal go to the directory where you have upload.py file then type 
#### streamlit run app.py



