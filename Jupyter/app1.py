from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoders
with open('bmi_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Label encoding dictionaries (if not using LabelEncoder)
label_encoding_dicts = {
    'Gender': {'Female': 0, 'Male': 1, 'Other': 2},
    'Activity Level': {'High': 0, 'Low': 1, 'Moderate': 2},
    'Health Issues': {
        'ADHD': 0, 'Asthma': 1, 'Critical Health Issue': 2, 'Obesity': 3,
        'Sport Injury/Accident': 4, 'nan': 5
    },
    'Emotional Well-being': {'Happy': 0, 'Normal': 1, 'Sad': 2},
    'Social Interaction': {'Daily': 0, 'Weekly': 1, 'nan': 2},
    'Cognitive Milestones': {'No': 0, 'Yes': 1},
    'Mental Health History': {'No': 0, 'Yes': 1},
    'Confidence and Self-esteem': {'Confused': 0, 'High': 1, 'Low': 2}
}

# Functions for suggestions based on prediction and user inputs
def provide_bmi_suggestions(category):
    suggestions = {
        "Underweight": [
            "Consider eating nutrient-dense foods such as nuts, avocados, and healthy oils.",
            "Increase your meal frequency by having smaller meals throughout the day.",
            "Incorporate resistance training to gain muscle mass.",
            "Track your meals to ensure you're meeting your calorie goals.",
            "Consult a dietitian for a personalized plan if needed."
        ],
        "Normal Weight": [
            "Great job maintaining a healthy weight!",
            "Keep up with a balanced diet and regular physical activity.",
            "Make sure you're drinking enough water daily.",
            "Continue practicing good sleep hygiene.",
            "Stay consistent with your habits to maintain long-term health."
        ],
        "Overweight": [
            "Try adding more fruits and vegetables to your meals.",
            "Aim to exercise at least 30 minutes a day, like brisk walking.",
            "Reduce processed foods and sugary beverages.",
            "Focus on portion control during meals.",
            "Track progress with non-scale victories (like energy levels!)."
        ],
        "Obesity": [
            "Consider working with a healthcare provider for support.",
            "Begin with low-impact exercises such as swimming or yoga.",
            "Set realistic, small goals for weight loss progress.",
            "Involve friends or family to stay motivated.",
            "Celebrate every small achievement along the way!"
        ]
    }
    return suggestions[category]

# Function to suggest height improvement for children aged 6 to 18
def height_suggestions(height, age, gender):
    # Ideal height ranges based on age and gender (example data)
    ideal_heights = {
        "male": {
            6: 115, 7: 120, 8: 125, 9: 130, 10: 135,
            11: 140, 12: 150, 13: 160, 14: 165, 15: 170,
            16: 175, 17: 178, 18: 180
        },
        "female": {
            6: 110, 7: 115, 8: 120, 9: 125, 10: 130,
            11: 140, 12: 150, 13: 160, 14: 165, 15: 168,
            16: 170, 17: 172, 18: 175
        }
    }

    suggestions = []  # Initialize an empty list for suggestions

    if gender.lower() in ideal_heights:
        ideal_height = ideal_heights[gender.lower()].get(age, 0)

        # Check if the user's height is below the ideal height
        if height < ideal_height:
            suggestions.append("Your height is below the ideal range for your age and gender.")
            suggestions.append("Here are some suggestions to help increase your height:")
            suggestions.append("- Ensure you get adequate nutrition, especially protein, calcium, and vitamin D.")
            suggestions.append("- Engage in stretching exercises like yoga or pilates.")
            suggestions.append("- Get enough sleep as growth hormone is released during sleep.")
            suggestions.append("- Maintain a good posture to appear taller.")
            suggestions.append("- Stay active with regular exercise to promote overall health.")
        else:
            suggestions.append("Your height is ideal for your age. Keep up your healthy habits!")
    
    return suggestions  # Ensure a list is returned

# Function to provide activity level suggestions
def provide_activity_suggestions(activity_level):
    suggestions = {
        "Low": [
            "Try to incorporate more physical activity into your daily routine.",
            "Consider joining a sport or physical activity group.",
            "Aim for at least 30 minutes of moderate exercise most days.",
            "Take the stairs instead of the elevator.",
            "Go for short walks during breaks or after meals."
        ],
        "Moderate": [
            "You're doing well! Try to maintain this level of activity.",
            "Incorporate strength training exercises twice a week.",
            "Explore new activities to keep your routine exciting.",
            "Make sure you're also balancing your nutrition.",
            "Consider setting new fitness goals to challenge yourself."
        ],
        "High": [
            "Excellent job staying active! Keep it up!",
            "Ensure you are fueling your body properly with nutrition.",
            "Consider mixing up your routine to prevent burnout.",
            "Make sure to include rest days to recover.",
            "Stay hydrated, especially during intense activities."
        ]
    }
    return suggestions.get(activity_level, [])

# Function to provide suggestions based on health issues
def provide_health_issue_suggestions(health_issue):
    suggestions = {
        "Obesity": [
            "Consider seeking guidance from a healthcare professional.",
            "Incorporate more physical activity into your daily routine.",
            "Focus on a balanced diet rich in fruits, vegetables, and whole grains.",
            "Monitor portion sizes and limit sugary snacks and drinks.",
            "Set realistic weight loss goals to stay motivated."
        ],
        "ADHD": [
            "Maintain a consistent routine for meals and activities.",
            "Consider consulting a healthcare professional for personalized strategies.",
            "Incorporate regular physical activity to help manage symptoms.",
            "Limit distractions during homework or study time.",
            "Ensure a balanced diet to support cognitive function."
        ],
        "Asthma": [
            "Avoid triggers that can worsen asthma symptoms.",
            "Consider consulting a healthcare provider for an action plan.",
            "Engage in physical activities that are suitable for asthma patients.",
            "Practice breathing exercises to improve lung capacity.",
            "Stay hydrated and avoid extreme temperatures."
        ],
        "Sport Injury/Accident": [
            "Consult a healthcare provider for rehabilitation advice.",
            "Incorporate low-impact exercises as you recover.",
            "Consider physical therapy to regain strength.",
            "Pay attention to any pain or discomfort and rest as needed.",
            "Ensure a balanced diet to support healing."
        ],
        "Critical Health Condition": [
            "Follow the advice of your healthcare provider closely.",
            "Maintain a balanced diet to support overall health.",
            "Stay informed about your condition and treatment options.",
            "Engage in physical activities approved by your healthcare provider.",
            "Seek support from family, friends, or support groups."
        ],
        "None": [
            "Great! Continue maintaining a healthy lifestyle with balanced nutrition and regular physical activity.",
            "Keep up with regular health check-ups to monitor your well-being.",
            "Engage in activities you enjoy to maintain physical and mental health."
        ]
    }
    return suggestions.get(health_issue, [])

# Function to provide suggestions based on emotional state
def provide_emotional_suggestions(emotional_state):
    suggestions = {
        "Happy": [
            "Keep spreading that positivity!",
            "Engage in activities that bring you joy.",
            "Share your happiness with friends and family.",
            "Consider journaling about what makes you happy.",
            "Stay active to maintain your good mood."
        ],
        "Sad": [
            "It's okay to feel sad sometimes; talk to someone about it.",
            "Engage in activities that you enjoy to lift your spirits.",
            "Consider practicing mindfulness or meditation.",
            "Spend time with friends or family who uplift you.",
            "Focus on self-care and take time for yourself."
        ],
        "Normal": [
            "Try to find activities that excite you.",
            "Consider exploring new hobbies or interests.",
            "Engage in social activities to connect with others.",
            "Reflect on what makes you feel good and pursue those activities.",
            "Maintain a balanced routine for emotional well-being."
        ]
    }
    return suggestions.get(emotional_state, [])

# Function to provide suggestions based on social interaction
def provide_social_suggestions(social_interaction):
    suggestions = {
        "Low": [
            "Consider joining clubs or groups that interest you.",
            "Reach out to friends or family for social activities.",
            "Engage in community events or volunteering.",
            "Try to connect with classmates or peers.",
            "Consider talking to a counselor for support."
        ],
        "Moderate": [
            "You're doing well! Keep engaging with your social circles.",
            "Try to maintain regular contact with friends.",
            "Consider exploring new social hobbies.",
            "Stay open to meeting new people.",
            "Reflect on your social interactions and adjust if needed."
        ],
        "Daily": [
            "Fantastic! Maintaining strong social connections is great for well-being.",
            "Continue engaging in activities with friends and family.",
            "Consider deepening existing relationships.",
            "Explore opportunities for community involvement.",
            "Balance social activities with time for yourself."
        ]
    }
    return suggestions.get(social_interaction, [])

# Function to provide suggestions based on cognitive milestones
def provide_cognitive_suggestions(cognitive_milestone):
    suggestions = {
        "On Track": [
            "Great job! Continue supporting your cognitive development.",
            "Engage in activities that challenge your brain.",
            "Consider puzzles, reading, or learning new skills.",
            "Maintain a balanced diet to support brain health.",
            "Stay active, as physical activity benefits cognitive function."
        ],
        "Struggling": [
            "Consider seeking support from teachers or tutors.",
            "Engage in activities that promote cognitive skills.",
            "Limit distractions during study time.",
            "Focus on regular review and practice.",
            "Consider seeking professional support if needed."
        ]
    }
    return suggestions.get(cognitive_milestone, [])

# Function to provide suggestions based on mental health history
def provide_mental_health_suggestions(mental_health_history):
    if mental_health_history == "Yes":
        return [
            "Consider seeking professional help to manage your mental health.",
            "Stay connected with supportive friends and family.",
            "Practice self-care activities that help you relax.",
            "Engage in physical activities to boost mood.",
            "Reflect on your feelings and consider journaling."
        ]
    else:
        return [
            "It's great to hear you have not had mental health issues.",
            "Continue prioritizing your mental well-being.",
            "Engage in activities that promote relaxation and joy.",
            "Maintain strong social connections for support.",
            "Consider practicing mindfulness or meditation."
        ]

# Function to provide suggestions based on confidence and self-esteem
def provide_confidence_suggestions(confidence):
    suggestions = {
        "Confused": [
            "Consider talking to a trusted friend or mentor for guidance.",
            "Set small, achievable goals to build confidence.",
            "Reflect on your strengths and accomplishments.",
            "Engage in activities that bring you joy.",
            "Consider seeking professional help if confusion persists."
        ],
        "High": [
            "Excellent! Maintain this positive self-image.",
            "Continue engaging in activities that challenge you.",
            "Consider helping others to build their confidence further.",
            "Keep setting new goals for yourself.",
            "Reflect on your achievements regularly."
        ],
        "Low": [
            "It's okay to feel low sometimes; consider talking to someone.",
            "Engage in activities that you enjoy to boost your mood.",
            "Reflect on your strengths and areas of success.",
            "Set small goals to help improve your self-esteem.",
            "Consider seeking support from a counselor or therapist."
        ]
    }
    return suggestions.get(confidence, [])
# Add other suggestion functions here...
# For simplicity, adding only BMI suggestions here. You can include others similarly.

# Route to render the input form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction and provide suggestions
# Route to handle prediction and provide suggestions
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
        prediction = model.predict(features)[0]

        # Get BMI suggestions based on the prediction
        bmi_suggestions = provide_bmi_suggestions(prediction)

        # Get height suggestions if the age is between 6 and 18
        if 6 <= age <= 18:
            height_suggestion = height_suggestions(height, age, gender)
        else:
            height_suggestion = ["Height suggestions are only available for children aged 6 to 18."]

        # Get various suggestions based on inputs
        activity_suggestions = provide_activity_suggestions(activity_level)
        health_issue_suggestions = provide_health_issue_suggestions(health_issues)
        emotional_suggestions = provide_emotional_suggestions(emotional_well_being)
        social_suggestions = provide_social_suggestions(social_interaction)
        cognitive_suggestions = provide_cognitive_suggestions(cognitive_milestones)
        mental_health_suggestions = provide_mental_health_suggestions(mental_health_history)
        confidence_suggestions = provide_confidence_suggestions(confidence)

        # Render the result page with BMI category and suggestions
        return render_template('result.html', 
                               prediction=prediction, 
                               bmi_suggestions=bmi_suggestions, 
                               height_suggestion=height_suggestion, 
                               activity_suggestions=activity_suggestions,
                               health_issue_suggestions=health_issue_suggestions,
                               emotional_suggestions=emotional_suggestions,
                               social_suggestions=social_suggestions,
                               cognitive_suggestions=cognitive_suggestions,
                               mental_health_suggestions=mental_health_suggestions,
                               confidence_suggestions=confidence_suggestions)

    except Exception as e:
        return render_template('error.html', error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True)
