import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

labs = ['Application mode', 'Application order', 'Course',
       'Daytime/evening attendance', 'Previous qualification',
       "Mother's qualification", "Father's qualification",
       "Mother's occupation", "Father's occupation", 'Displaced', 'Debtor',
       'Tuition fees up to date', 'Gender', 'Scholarship holder',
       'Age at enrollment', 'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
       'Inflation rate', 'GDP']

cat_labs = ['Application mode', 'Course',
       'Daytime/evening attendance', 'Previous qualification',
       "Mother's qualification", "Father's qualification",
       "Mother's occupation", "Father's occupation", 'Displaced', 'Debtor',
        'Gender', 'Scholarship holder']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]

    input_data = pd.DataFrame([int_features], columns=labs)
    input_df_encoded = pd.get_dummies(input_data, columns=cat_labs)
    ohedf = pd.read_csv('OHEdata.csv')
    del ohedf[ohedf.columns[0]]

    input_df_encoded = input_df_encoded.reindex(columns=ohedf.columns, fill_value=0)

    input_df_encoded = input_df_encoded.drop(columns=['Target'], errors='ignore')
    

    prediction = model.predict(input_df_encoded) 
    result = prediction[0]

    return render_template('results.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)