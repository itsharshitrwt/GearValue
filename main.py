from flask import Flask , render_template , request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open("models/vehicleRegressionModel.pkl","rb"))
car = pd.read_csv("data/Cleaned Car.csv")

model1 = pickle.load(open("models/bikeRegressionModel.pkl","rb"))
bike = pd.read_csv("data/new_bikes_data1.csv")


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/cars")
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique() , reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0, "Select Company")
    return render_template('index.html' , companies = companies, car_models= car_models, years = year, fuel_types= fuel_type)

@app.route('/predict', methods= ['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = request.form.get('kilo_driven')
    print(company , car_model , year , fuel_type, kms_driven)

    prediction = model.predict(pd.DataFrame([[car_model, company , year, kms_driven, fuel_type]] , columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    
    return str(np.round(prediction[0], 2))


@app.route("/bikes")
def index1():
    companies = sorted(bike['brand'].unique())
    bike_models = sorted(bike['bike_name'].unique())
    owner = bike['owner'].unique()
    companies.insert(0, "Select Company")
    return render_template('index1.html' , companies = companies, bike_models = bike_models, owners = owner) 

@app.route('/predict1', methods= ['POST'])
def predict1():
    company = request.form.get('company')
    bike_model = request.form.get('bike_model')
    owner = request.form.get('owner')
    kms_driven = request.form.get('kilo_driven')
    print(company , bike_model ,owner , kms_driven)

    prediction = model1.predict(pd.DataFrame([[bike_model, company ,kms_driven,owner]] , columns=['bike_name', 'brand', 'kms_driven', 'owner']))
    
    return str(np.round(prediction[0], 2))



if __name__ == "__main__":
    app.run(debug = True)