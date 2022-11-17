from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

df_col_names = pickle.load(open('df_col_names.pkl', 'rb'))
ohe_df_col_names = pickle.load(open('ohe_df_col_names.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    form_values = request.form.values()
    input_data = pd.DataFrame(dict(zip(df_col_names, form_values)), index=[0])
    ohe_input = pd.get_dummies(input_data).reindex(columns = ohe_df_col_names, fill_value=0)
    ohe_input = scaler.transform(ohe_input)
    prediction = model.predict(ohe_input)

    if int(prediction)==0:
        return render_template('result.html', prediction_text='We are sorry to inform that you may have high mortality Diabetes! Please seek medical advise')                 
    else:
        return render_template('result.html', prediction_text='Congratulations! You dont have high mortality with Diabetes! Stay Healthy')



if __name__ == "__main__":
    app.run(debug=False)