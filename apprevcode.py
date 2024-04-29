import os
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open(os.path.join("D:/template_reversecode","svm_clf.pkl"), "rb"))

# Set the template folder path
template_folder = os.path.join("D:/template_reversecode", "templates")
app.template_folder = template_folder

# Set the static folder path
static_folder = os.path.join("D:/template_reversecode", "static")
app.static_folder = static_folder

# Mapping dictionaries for reverse encoding
smoking_map = {0: 'Non-Smoker', 1: 'Former Smoker', 2: 'Current Smoker'}
stroke_map = {0: 'No', 1: 'Yes'}
diff_walking_map = {0: 'No Difficulty', 1: 'Some Difficulty', 2: 'Severe Difficulty'}
sex_map = {0: 'Male', 1: 'Female'}
age_category_map = {0: '<45', 1: '45-54', 2: '55-64', 3: '65-74', 4: '>=75'}
race_map = {0: 'White', 1: 'Black', 2: 'Other'}
diabetic_map = {0: 'No', 1: 'Yes'}
physical_activity_map = {0: 'No', 1: 'Yes'}
gen_health_map = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Very Good', 4: 'Excellent'}
asthma_map = {0: 'No', 1: 'Yes'}
kidney_disease_map = {0: 'No', 1: 'Yes'}
skin_cancer_map = {0: 'No', 1: 'Yes'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get feature values from the form
    bmi = float(request.form['BMI'])
    smoking = smoking_map[int(request.form['Smoking'])]
    alcohol_drinking = request.form['AlcoholDrinking']
    stroke = stroke_map[int(request.form['Stroke'])]
    physical_health = float(request.form['PhysicalHealth'])
    mental_health = float(request.form['MentalHealth'])
    diff_walking = diff_walking_map[int(request.form['DiffWalking'])]
    sex = sex_map[int(request.form['Sex'])]
    age_category = age_category_map[int(request.form['AgeCategory'])]
    race = race_map[int(request.form['Race'])]
    diabetic = diabetic_map[int(request.form['Diabetic'])]
    physical_activity = physical_activity_map[int(request.form['PhysicalActivity'])]
    gen_health = gen_health_map[int(request.form['GenHealth'])]
    sleep_time = float(request.form['SleepTime'])
    asthma = asthma_map[int(request.form['Asthma'])]
    kidney_disease = kidney_disease_map[int(request.form['KidneyDisease'])]
    skin_cancer = skin_cancer_map[int(request.form['SkinCancer'])]

    # Make prediction
    prediction = model.predict([[bmi, smoking, alcohol_drinking, stroke, physical_health, mental_health, diff_walking, sex, age_category, race, diabetic, physical_activity, gen_health, sleep_time, asthma, kidney_disease, skin_cancer]])[0]

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    # Get the port number from the environment variable PORT or use 4000 as fallback
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port)
