from flask import Flask,jsonify,request
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv(r"D:\Velocity\Machine Learning Practice\Bengaluru_House_Data.csv")
pune_knn=pickle.load(open("Pune_house_regressor.pkl","rb"))

app=Flask(__name__)

@app.route('/')
def main():
    return jsonify({"message":"Active"})



@app.route('/house_price')
def house_price():
    data=request.get_json()
    availability=data["availability"]
    size=data["size"]
    total_sqft=data["total_sqft"]
    bath=data["bath"]
    balcony=data["balcony"]
    
    d={"availability":[availability],"size":[size],"total_sqft":[total_sqft], "bath":[bath], "balcony":[balcony] }
    input=pd.DataFrame(data=d)

    prediction=pune_knn.predict(input)

    return jsonify({"availability":availability,"size":size,"total_sqft":total_sqft, "bath":bath,
                     "balcony":balcony, "prediction":prediction[0]})

if __name__=="__main__":
    app.run(debug=True)