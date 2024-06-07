import matplotlib.cm as cm
#from IPython.display import Image, display
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display
from keras.layers import concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import cohen_kappa_score,roc_auc_score,confusion_matrix,classification_report
import streamlit as st
from io import BytesIO

gru_model = tf.keras.models.load_model("gru_model.h5")


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



classes = ["COPD" ,"Bronchiolitis ", "Pneumoina", "URTI", "Healthy"]


def add_noise(data,x):
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise

def shift(data,x):
    return np.roll(data, x)

def stretch(data, rate):
    data = librosa.effects.time_stretch(data, rate=rate)
    return data

def pitch_shift (data , rate):
    data = librosa.effects.pitch_shift(data, sr=220250, n_steps=rate)
    return data



def gru_diagnosis_prediction(test_audio_path):
    data_x, sampling_rate = librosa.load(test_audio_path)
    data_x = stretch(data_x, 1.2)

    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T, axis=0)
    features = features.reshape(1, 52)

    test_pred = gru_model.predict(np.expand_dims(features, axis=1))
    classpreds = classes[np.argmax(test_pred[0], axis=1)[0]]
    confidence = test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()

    return classpreds, confidence

# Function to save uploaded file to temporary location and call prediction function
def process_uploaded_file(uploaded_file):
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    type, accu = gru_diagnosis_prediction("temp.wav")
    os.remove("temp.wav")
    return type, accu


new_title = '<p style="font-family:sans-serif;  font-size: 42px; color:black">RESPIRATORY DISEASE DETECTION BASED ON LUNG SOUNDS USING DEEP LEARNING</p>'
st.markdown(new_title, unsafe_allow_html=True)
### load file
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav","audio"])
if uploaded_file is not None:
    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        type,accu= process_uploaded_file(uploaded_file)
        st.markdown("<span style='font-size:30px;color:black'>The given Lung Sound is of type : {}</span>".format(type),unsafe_allow_html=True)
        st.markdown("<span style='font-size:30px;color:black'>The chances of Lung Sound being {} is: {}</span>".format(type,accu), unsafe_allow_html=True)