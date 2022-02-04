import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
model = pickle.load(
    open('gender_model.pkl', 'rb'))


@app.route('/')
def home():
    return predict()


def getPredicitedCategorties():
    test_data = pd.read_csv("s3://capstonemlc/deployment/test_data.csv")
    test_data=test_data.sample(n = 50)
    test_data.reset_index(drop=True, inplace=True)
    test_data.drop("Unnamed: 0", axis=1, inplace=True)

    #get X_test, y_test from test_data
    X_test = test_data.loc[:, ((test_data.columns != 'device_id') & (test_data.columns != 'age') & (test_data.columns != 'gender'))]
    y_test = test_data.loc[:, (test_data.columns == 'gender')]

    model = pickle.load(open('gender_model.pkl', 'rb'))
    result = model.predict_proba(X_test)

    zeros = [i[0] for i in result]
    ones  = [i[1] for i in result]

    test_pred = pd.DataFrame(ones)
    test_df = pd.DataFrame(y_test)
    test_final = pd.concat([test_df, test_pred],axis=1,ignore_index=True)

    test_final= test_final.rename(columns={ 0 : 'gender',1 : 'gender_prob'})
    test_final['final_prediction'] = test_final.gender_prob.map(lambda x: 1 if x >= 0.6251 else 0)

    test_final['Campain 1'] = test_final.gender_prob.map(lambda x: 1 if x > 0.7340 else 0)
    test_final['Campain 2'] = test_final.gender_prob.map(lambda x: 1 if x < 0.5678 else 0)
    test_final['Campain 3'] = test_final.gender_prob.map(lambda x: 1 if x < 0.5678 else 0)
    
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
