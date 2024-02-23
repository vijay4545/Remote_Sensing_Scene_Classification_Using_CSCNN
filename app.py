import streamlit as st
import numpy as np
from numpy import asarray
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tensorflow as tf
# import numpy as np
from tensorflow.keras.preprocessing import image
model = load_model("model_sequential.h5")

image_size = (256, 256, 3)
batch_size = 32


def predict(image_path):
    class_labels = {
        'aGrass': 0,
        'bField': 1,
        'cIndustry': 2,
        'dRiverLake': 3,
        'eForest': 4,
        'fResident': 5,
        'gParking': 6}
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    # class_labels = train_generator.class_indices
    predicted_class_label = list(class_labels.keys())[predicted_class_index]

    # print('Predicted class:', predicted_class_label)
    return predicted_class_label


st.title('Remote sensing Scene Classification')


imge = st.file_uploader('Upload your file', type=['JPG', 'PNG', 'JPEG', 'TIFF'], accept_multiple_files=False, key=None, help=None,
                        on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


if (imge != None):
    st.image(imge, caption='Uploaded Image')

if st.button('Predict'):

    predict = predict(imge)
    st.markdown(""" <style> .predict {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)
    st.write(predict)
