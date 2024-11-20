from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoders
with open('bmi_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Label encoding dictionaries (if not using LabelEncoder)
label_encoding_dicts = {
    'Gender': {
        'Female': 0,
        'Male': 1,
        'Other': 2
    },
    'Activity Level': {
        'High': 0,
        'Low': 1,
        'Moderate': 2
    },
    'Health Issues': {
        'ADHD': 0,
        'Asthma': 1,
        'Critical Health Issue': 2,
        'Obesity': 3,
        'Sport Injury/Accident': 4,
        'nan': 5
    },
    'Emotional Well-being': {
        'Happy': 0,
        'Normal': 1,
        'Sad': 2
    },
    'Social Interaction': {
        'Daily': 0,
        'Weekly': 1,
        'nan': 2
    },
    'Cognitive Milestones': {
        'No': 0,
        'Yes': 1
    },
    'Mental Health History': {
        'No': 0,
        'Yes': 1
    },
    'Confidence and Self-esteem': {
        'Confused': 0,
        'High': 1,
        'Low': 2
    }
}

# Route to render the input form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        gender = request.form['gender']
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        activity_level = request.form['activity_level']
        health_issues = request.form['health_issues']
        emotional_well_being = request.form['emotional_well_being']
        social_interaction = request.form['social_interaction']
        cognitive_milestones = request.form['cognitive_milestones']
        mental_health_history = request.form['mental_health_history']
        confidence = request.form['confidence']

        # Manually map the categorical values using the dictionaries
        gender_encoded = label_encoding_dicts['Gender'][gender]
        activity_level_encoded = label_encoding_dicts['Activity Level'][activity_level]
        health_issues_encoded = label_encoding_dicts['Health Issues'][health_issues]
        emotional_well_being_encoded = label_encoding_dicts['Emotional Well-being'][emotional_well_being]
        social_interaction_encoded = label_encoding_dicts['Social Interaction'][social_interaction]
        cognitive_milestones_encoded = label_encoding_dicts['Cognitive Milestones'][cognitive_milestones]
        mental_health_history_encoded = label_encoding_dicts['Mental Health History'][mental_health_history]
        confidence_encoded = label_encoding_dicts['Confidence and Self-esteem'][confidence]

        # Prepare the feature array for prediction
        features = np.array([[age, gender_encoded, height, weight, activity_level_encoded, 
                              health_issues_encoded, emotional_well_being_encoded, 
                              social_interaction_encoded, cognitive_milestones_encoded,
                              mental_health_history_encoded, confidence_encoded]])

        # Predict using the loaded model
        prediction = model.predict(features)

        # Return the result
        return render_template('result.html', prediction=prediction[0])
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
