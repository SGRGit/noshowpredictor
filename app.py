import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime

app = Flask(__name__)
model = pickle.load(open('model_f.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index_f.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on PHP GUI
    '''
    #int_features = np.zeros((1, 10))
    #int_features =[int(x) for x in request.form.values()]
    age = request.values.get("Age")

    sms = request.values.get("Sms Received")
    
    if request.values.get("Appointment Date") == None:
        appdt = datetime.strptime('1900-01-01', '%Y-%m-%d').date()
    else:
        appdt = datetime.strptime(request.values.get("Appointment Date"), '%Y-%m-%d').date()
    
    if request.values.get("Scheduled Date") == None:
        schdt = datetime.strptime('1900-01-01', '%Y-%m-%d').date()
    else:
        schdt = datetime.strptime(request.values.get("Scheduled Date"), '%Y-%m-%d').date()
    
    deltday = abs((appdt - schdt).days)
    inp = np.array([age, 0, 0, 0, 0, sms, 0, deltday, 0, 0]).reshape(1, 10)
  
    prediction = round(model.predict_proba(inp)[0][1] *100, 2)

    return render_template('index_f.html', prediction_text = 'Appointment Chance {} %'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)