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
    #choose only the most important features to test (everything else stays same) based on exploratory data analysis
    int_features = [float(x) for x in request.form.values()]

    #hard-coded data for the demo
    arr = [1, 4, 12, 0, 1, 13, 27, 10, 8, 1, 0, 1, 1, 0, 18, 0, 6, 9, 6, 12.2, 0, 0, 6, 6, 6, 13.5, 0, 12.7, 1.4, -1.7]
    #tuition fees
    arr[11] = int_features[0]
    #scholarship holder
    arr[13] = int_features[1]
    # 1st sem enrolled
    arr[16] = int_features[2]
    #1st sem approved
    arr[18] = int_features[3]
    #2nd sem enrolled
    arr[22] = int_features[4]
    #2nd sem evaluations
    arr[23] = int_features[5]
    #2nd sem approved
    arr[24] = int_features[6]
    #unemployment rate
    arr[28] = int_features[7]


    #one-hot encoding of data to fit model
    input_data = pd.DataFrame([arr], columns=labs)
    input_df_encoded = pd.get_dummies(input_data, columns=cat_labs)
    ohedf = pd.read_csv('OHEdata.csv')
    del ohedf[ohedf.columns[0]]

    input_df_encoded = input_df_encoded.reindex(columns=ohedf.columns, fill_value=0)

    input_df_encoded = input_df_encoded.drop(columns=['Target'], errors='ignore')
    

    prediction = model.predict(input_df_encoded) 
    result = prediction[0]


    #map number to string

    mapping = {
    0: "Dropout",
    1: "Graduate",
    2: "Enrolled",
    }

    result_str = mapping.get(result, "Unknown") 

    return render_template('results.html', prediction=result_str)



if __name__ == "__main__":
    app.run(debug=True)