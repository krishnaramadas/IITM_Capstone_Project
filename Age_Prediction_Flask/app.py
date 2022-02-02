import numpy as np
import pandas as pd
import s3fs
from flask import Flask, request, jsonify, render_template
import pickle
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
model = pickle.load(
    open('age_model.pkl', 'rb'))

@app.route('/')
def home():
    return predict()

def getPredicitedCategorties():
    test_data = pd.read_csv("s3://capstonemlc/deployment/test_data.csv")
    test_data=test_data.drop(['Unnamed: 0'], axis=1)
    #test_data=test_data.sample(n = 50)
    test_data=test_data.iloc[:50, :]
    test_data['constant']=1
    X_test = test_data.loc[:, ((test_data.columns != 'device_id') & (test_data.columns != 'age'))]
    y_test = test_data.loc[:, (test_data.columns == 'age')]
    
    model = pickle.load(open('age_model.pkl', 'rb'))
    result = model.predict(X_test)
    
    test_final = pd.DataFrame({'age':y_test.values.reshape(-1), 'age_pred':result})
    test_final['age_pred']=test_final['age_pred'].astype('int')
    
    test_final['Campaign 4'] = test_final.age_pred.map(lambda x: 1 if x <=24 else 0)
    test_final['Campaign 5'] = test_final.age_pred.map(lambda x: 1 if ((x >24) & (x<=32)) else 0)
    test_final['Campaign 6'] = test_final.age_pred.map(lambda x: 1 if x>32 else 0)
    
    test_data_deviceId = test_data.device_id.to_frame()

    test_data_final = test_data_deviceId.join(test_final, how='left')
    test_data_final=test_data_final.reset_index()
    test_data_final=test_data_final.drop(['index'], axis=1)
    return test_data_final


@app.route('/predict', methods=['POST'])
def predict():

    predicted_Values = getPredicitedCategorties()
    table = predicted_Values.to_html(classes='table table-striped table-bordered')
    return render_template('index.html', tables=[table], titles='')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
