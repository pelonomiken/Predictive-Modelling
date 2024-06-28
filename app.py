from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open('knnc.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['Age'])
    course = request.form['Course']
    gender = request.form['Gender']
    cgpa = float(request.form['CGPA'])
    sleep_quality = request.form['Sleep Quality']
    physical_activity = request.form['Physical Activity']
    diet_quality = request.form['Diet Quality']
    social_support = request.form['Social Support']
    substance_use = request.form['Substance Use']
    financial_stress = float(request.form['Financial Stress'])
    emotional_distress = float(request.form['Emotional Distress'])

    
    course_dict = {'Others': 5, 'Engineering': 2, 'Business': 0, 'Computer': 1, 'Medical': 4, 'Law': 3}
    gender_dict = {'Female': 0, 'Male': 1}
    sq_dict = {'Average': 0, 'Good': 1, 'Poor': 2}
    pa_dict = {'High': 0, 'Low': 1, 'Moderate': 2}
    dq_dict = {'Average': 0, 'Good': 1, 'Poor': 2}
    ss_dict = {'High': 0, 'Low': 1, 'Moderate': 2}
    su_dict = {'Frequently': 0, 'Never': 1, 'Occasionally': 2}


    course_numeric = course_dict.get(course, 0)
    gender_numeric = gender_dict.get(gender, 0)
    sleep_quality_numeric = sq_dict.get(sleep_quality, 0)
    physical_activity_numeric = pa_dict.get(physical_activity, 0)
    diet_quality_numeric = dq_dict.get(diet_quality, 0)
    social_support_numeric = ss_dict.get(social_support, 0)
    substance_use_numeric = su_dict.get(substance_use, 0)


   
    input_data = np.array([[age, course_numeric, gender_numeric, cgpa, sleep_quality_numeric,
                            physical_activity_numeric, diet_quality_numeric, social_support_numeric, substance_use_numeric,
                            financial_stress, emotional_distress]])
    prediction = model.predict(input_data)

   
    prediction_text = f"The predicted output is: {prediction}"

    
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
