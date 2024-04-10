# Assuming you have a function named find_index() that takes a string argument

from fastapi import FastAPI, Request, Body, Query
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re
import re
import random

# TensorFlow and Keras imports



# Optional imports for data manipulation and preprocessing (uncomment if needed)
import numpy as np
import pandas as pd
import json

# Optional import for text classification (uncomment if needed)
from sklearn.preprocessing import LabelEncoder

# Optional import for serialization (uncomment if needed)
import pickle
uploaded = "api/chatbot.json"

with open("api/chatbot.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

# Almacenar datos que serán convertidos a DataFrame
dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]

    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

# Crear un nuevo DataFrame a partir del diccionario
df = pd.DataFrame.from_dict(dic)

# Mostrar el DataFrame para verificar
df.head()

# Obtener las etiquetas únicas
df['tag'].unique()
# Load the tokenizer, label encoder, and trained model
with open('api/tokenizer_lstm.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('api/label_encoder_lstm.pkl', 'rb') as handle:
    lbl_enc = pickle.load(handle)

# Replace 'my_model.keras' with the actual filename of your saved model
model = load_model('api/my_lstm_model.keras')  # (LSTM, BiLSTM, GRU, or BiGRU)

def input_user(pattern):
    """
    Preprocesses user input for model prediction.

    Args:
        pattern (str): The user's input text.

    Returns:
        list: A list containing the preprocessed user input sequence.
    """

    text = []
    txt = re.sub(r"[^a-zA-Z\']", ' ', pattern)  # Remove non-alphanumeric characters
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    # Tokenize and pad the input sequence
    x_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(x_test, padding='post', maxlen=18)  # Adjust maxlen if needed

    return x_test

def predict(pattern):
    """
    Predicts the chatbot's response to the user's input.

    Args:
        pattern (str): The user's input text.

    Returns:
        str: The predicted response from the chatbot.
    """

    x_test = input_user(pattern)
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=1)
    tag = lbl_enc.inverse_transform(y_pred)[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    respuesta=random.choice(responses)
    return  respuesta 


app = FastAPI()

@app.get("/api/lala")
async def handle_lala(dato: str = Query(...)):
  # Implementar la lógica de la función usando el valor de `dato`
    resultado = predict (dato)

    return {"resultado": resultado}

