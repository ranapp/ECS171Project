import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
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
       'Inflation rate', 'GDP', 'Target']
    
    cat_labs = ['Application mode', 'Course',
       'Daytime/evening attendance', 'Previous qualification',
       "Mother's qualification", "Father's qualification",
       "Mother's occupation", "Father's occupation", 'Displaced', 'Debtor',
        'Gender', 'Scholarship holder']
    
    cats = ['Marital status',	'Application mode', 'Daytime/evening attendance',	'Previous qualification', 'Nacionality',	"Mother's qualification",	"Father's qualification",	"Mother's occupation", \
         "Father's occupation", 'Debtor', 'Gender',	'Scholarship holder', 'Course', 'International','Tuition fees up to date', 'Educational special needs','Displaced'] 
    column_names = df_ohe.columns.tolist()
    cats_set = set(cats)
    names_set = set(column_names)

    cats = names_set.intersection(cats_set)
    print("Unique elements:")
    for cat in cats:
        print(df_ohe[cat].unique()) 
    
    input_df = pd.DataFrame([int_features], columns=labs)
    input_df_encoded = pd.get_dummies(input_df, columns=cat_labs)
    input_df_encoded = input_df_encoded.reindex(columns=labs[:-1], fill_value=0)

    column_names = df_ohe.columns.tolist()

    common_strings = names_set.intersection(cats_set)

    # numerical attributes
    nats = [item for item in names_set if item not in common_strings]
    nats.remove('Target')





    prediction = model.predict(features) 
    result = prediction[0]

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)