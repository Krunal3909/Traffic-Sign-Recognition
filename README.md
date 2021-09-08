[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) 

# Traffic Sign Classifier using Streamlit

## Project overview
- The aim of this project is to focus on the first fundamental features of the decision making ability of an autonomous vehicle, 
i.e., to develop a deep learning model that reads traffic signs and classifies them correctly using Convolutional Neural Networks(CNNs).

- The traffic sign classifier uses a German traffic dataset. The German traffic dataset consists of
34,799 32*32 pixels colored images that is used for the training dataset, 12,630 images are used
for the testing dataset and 4410 images are used in the validation dataset where each images is a
photo of a traffic sign belonging to one of the 36 classes i.e., traffic sign types.


## Dataset:
Dataset can be found here : https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign


## Folders Description
### Model.py File
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

### Result Excel
- Conatins a exccel sheet having the results of our test results on random images from the internet
- Also contains the accuracy of our model on unseen data
- Accuracy on unseen data : 79.06%

### Model
Contains the saved keras model named
- ###### TSR.hdf5

## Files for our Streamlit App


#### app.py
- Contains the front-end code for the streamlit app.
- Imports the predict() function fetches the result and displays it.
- get_model(): Loads the saved model into cache using streamlit's "@st.cache" feature.
- predict(): Takes an image as input from the function parameter, preprocesses it and feeds it to the model for results.

## App Output
#### First Page
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/WebApp%20Images/Front%20Page.png)
#### Image upload page
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/WebApp%20Images/Image%20upload.png)
#### Result Page
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/WebApp%20Images/Result.png)

## Model Output
#### Summary of CNN Model
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/Model%20Results%20Images/model_summary.png)
#### Accuracy Graph
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/Model%20Results%20Images/accuracy_chart.png)
#### Loss Graph
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/Model%20Results%20Images/loss_chart.png)
#### Final prediction
![alt text](https://github.com/Krunal3909/Traffic-Sign-Recognition/blob/main/Model%20Results%20Images/final_prediction.png)


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
- In your Command line/Terminal go to the directory where you have upload (.py) file then type 
#### streamlit run app.py



