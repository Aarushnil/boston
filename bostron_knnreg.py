from flask import Flask, jsonify,request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
df=pd.read_csv("boston_df.csv")
# df_dev=pd.read_csv('dev.csv')
# df_mean=pd.read_csv('mean.csv')

linear_model=pickle.load(open("boston.pkl",'rb'))


@app.route('/')
def main():
    return jsonify({"message":"active"})

@app.route('/boston')
def boston():
    data=request.get_json()
    CRIM=data['CRIM']
    ZN=data['ZN']
    INDUS= data['INDUS']
    CHAS= data['CHAS']
    NOX=data['NOX'] 
    RM=data['RM'] 
    AGE=data['AGE']
    DIS= data['DIS'] 
    RAD=data['RAD']
    TAX=data['TAX']
    PTRATIO=data['PTRATIO']
    B=data['B']
    LSTAT=data['LSTAT']

    d={'CRIM':[CRIM], 'ZN':[ZN], 'INDUS':[INDUS], 'CHAS':[CHAS], 'NOX':[NOX], 'RM':[RM], 'AGE':[AGE], 'DIS':[DIS],
     'RAD':[RAD], 'TAX':[TAX], 'PTRATIO':[PTRATIO], 'B':[B], 'LSTAT':[LSTAT]}
     
    input=pd.DataFrame(data=d)
    

    prediction=linear_model.predict(input)

    return jsonify({"prediction":prediction[0]})

if __name__=="__main__":
    app.run(debug=True)