# panggil library yang dibutuhkan
import flask
import pandas as pd
import tensorflow as tf
import keras
import pickle 
import numpy as np
from keras.models import load_model

# memulai app flask
app = flask.Flask(__name__)


# load model yang sudah dilatih beserta file label nya. 
# inisiasi graph agar bisa 
global graph
graph = tf.get_default_graph()
model = load_model('iris_model.h5')
file = open('label.pickle', 'rb').read()
lb = pickle.loads(file)

# membuat fungsi prediksi dengan memasukan variabel masukannya
@app.route("/predict", methods=["GET","POST"])
def predict():
    # awali kondisi nya 0 atau false
    data = {"success": False}

    # inisialisasi variabel untuk menyimpan file request dalam format json
    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # jika ada request yang masuk, maka aktifkan fungsi prediksi (return prediksi)
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(lb.classes_[model.predict(x).argmax(1)[0]])
            data["score"] = str(np.max(model.predict(x)*100))
            data["success"] = True

    # return nilai data dalam format json
    return flask.jsonify(data)    

# menjalankan web server 
app.run(debug=True)

# contoh url untuk request prediksi
# http://127.0.0.1:5000/predict?sepal_length=1.4&sepal_width=3.4&petal_length=5.4&petal_width=1.4
# http://127.0.0.1:5000/predict?sepal_length=4.0&sepal_width=4.4&petal_length=4.4&petal_width=2.2
