from cryptography.fernet import Fernet
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
import tensorflow_text

DIR = './'


@st.cache(allow_output_mutation=True)
def load_model():
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    predict = interpreter.get_signature_runner('serving_default')
    return lambda x: predict(input_1=np.expand_dims([x], 0))['output_1']


model = load_model()

fernet = Fernet(SECRET_KEY)

# opening the encrypted file
with open('class_encrypted.json', 'rb') as enc_file:
    encrypted = enc_file.read()

# decrypting the file
classes = json.loads(fernet.decrypt(encrypted))

st.set_page_config(page_title='NLP', layout='centered')

st.title('NLP')

slider = st.sidebar.slider('Confidence Threshold (default 80%)', 0, 100, 80)

oem = st.sidebar.selectbox('Select Name of OEM', ['None', 'Porsche', 'Lexus', 'Honda', 'Toyota', 'Ford', 'Chrysler', 'Dodge',
                                                  'Jeep', 'Ram', 'Chevrolet', 'BMW', 'Subaru', 'Audi', 'Volkswagen',
                                                  'Mazda', 'Volvo', 'Kia', 'Nissan', 'Mercedes-Benz', 'Lincoln',
                                                  'Hyundai', 'Acura', 'Jaguar', 'Land Rover', 'GMC', 'Buick',
                                                  'INFINITI', 'Cadillac', 'FIAT', 'Alfa Romeo', 'Mitsubishi',
                                                  'Genesis', 'MINI', 'Maserati'])

features = st.sidebar.text_input('Enter feature names. (Delimited by |)')

if features:
    features = features.split('|')

    for feature in features:

        result = model(oem + ' ' + feature if oem != 'None' else feature)
        prob = np.max(result)
        label = classes[np.argmax(result)]

        if prob > slider/100:
            st.success(f'{feature} --- {label} ---- {prob}')
        else:
            st.warning(f'{feature} --- {label} ---- {prob}')



