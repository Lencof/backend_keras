from flask import Flask
from flask import Flask, request
from flask_restful import Resource, Api
import sqlalchemy
from json import dumps
from flask_jsonpify import jsonify
from keras.models import load_model
from keras import backend as K
from os import environ
from keras.preprocessing import sequence
import numpy
import re
import pickle
from sys import stdin
import json
from langdetect import detect

def load_obj(name):
    with open('' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

app = Flask(__name__)
api = Api(app)
db_connect = sqlalchemy.create_engine('sqlite:////home/sovietspy2/PycharmProjects/backend_keras/database.db')
model = load_model("/home/sovietspy2/PycharmProjects/backend_keras/model2.h5")
max_sentence_length = 200
vocab_to_int = load_obj('/home/sovietspy2/PycharmProjects/backend_keras/vocab_to_int')
int_to_languages = load_obj('/home/sovietspy2/PycharmProjects/backend_keras/int_to_languages')
@app.after_request

@app.route('/')
def hello_world():
    return 'backend api running'

def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response


def to_long_lang(text):
    if text=='en':
        return 'english'
    elif text=='de':
        return 'german'
    elif text=='fr':
        return 'french'
    elif text=='hu':
        return 'hungarian'
    else:
        return 'err'


class Prediction(Resource):
    def get(self):
        conn = db_connect.connect() # connect to database
        query = conn.execute("select * from prediction")
        result = query.cursor.fetchall()
        print(result)

        rows = []

        for i in range(0,len(result)):
            nn = list(result[i])
            #rows.append(
            print(nn)
            rows.append({ 'id':nn[0], 'text':nn[1], 'prediction': nn[2], 'valid pred': nn[3] })

        data = {
            'data': rows
        }

        return jsonify(data)


class Predict(Resource):
    def get(self, text):
        if len(text)!=0:
            ret = predict_sentence(text)

            result = {'data': {
                'text': text,
                'language': ret,
            }}

            conn = db_connect.connect()
            conn.execute("insert into prediction (text, predicted, actual) VALUES ('{0}', '{1}', NULL)".format(text, ret))

        else:
            result = {
                'data': 'no text'
            }
        return jsonify(result)


class ValidPredict(Resource):
    def get(self, text):
        if len(text)!=0:
            ret = predict_sentence(text)
            valid = detect(text)

            result = {'data': {
                'text': text,
                'language': ret,
                'valid': to_long_lang(valid),
            }}

            conn = db_connect.connect()
            conn.execute("insert into prediction (text, predicted, actual) VALUES ('{0}', '{1}', '{2}')".format(text, ret,to_long_lang(valid)))


        else:
            result = {
                'data': 'no text'
            }
        return jsonify(result)

api.add_resource(Prediction, '/list')
api.add_resource(Predict, '/p/<text>')
api.add_resource(ValidPredict, '/pvalid/<text>')

def convert_to_int(data, data_int):

    all_items = []
    for sentence in data:
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in str(sentence).split()])

    return all_items

def process_sentence(sentence):

    return re.sub('[^A-Za-z0-9]+', ' ', sentence).lower().strip()

def predict_sentence(sentence):

    sentence = process_sentence(sentence)

    x = numpy.array(convert_to_int([sentence], vocab_to_int))
    x = sequence.pad_sequences(x, maxlen=max_sentence_length)

    prediction = model.predict(x)
    #print(int_to_languages)
    #print(prediction)

    lang_index = numpy.argmax(prediction)

    return int_to_languages[lang_index]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
