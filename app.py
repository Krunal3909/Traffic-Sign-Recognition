import streamlit as st 
from PIL import Image,ImageOps
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def get_model():
        model = load_model('TSR.hdf5')
        print('Model Loaded')
        return model 
model =get_model()
st.write("Traffic Sign Classifier")
file = st.file_uploader("Please upload an image of Traffic Sign", type=["png", "jpg"])

def import_and_predict(image_data, model):
    
        
        image= ImageOps.fit(image_data,(30,30),Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("please upload a file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    if st.button('predict'):
        st.write("Result...")
        predictions = import_and_predict(image, model)
        sign_names = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
            }
        string = "This sign is of :"+sign_names[np.argmax(predictions)]
        st.success(string)
    
