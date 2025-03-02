from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import ast

# Charger les données
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Convertir les chaînes en listes Python
medications['Medication'] = medications['Medication'].apply(ast.literal_eval)
diets['Diet'] = diets['Diet'].apply(ast.literal_eval)

# Charger le modèle
svc = pickle.load(open("models/svc.pkl", 'rb'))

app = Flask(__name__)

# ============================================================
# Fonctions personnalisées et utilitaires
# ========================== Fonctions utilitaires ============
def helper(dis):
    # Description
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    # Precautions
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    # Medications
    med = medications[medications['Disease'] == dis]['Medication'].tolist()
    if med:  # Si la liste n'est pas vide
        med = med[0]  # Extraire la première liste (car tolist() renvoie une liste de listes)

    # Diet
    die = diets[diets['Disease'] == dis]['Diet'].tolist()
    if die:  # Si la liste n'est pas vide
        die = die[0]  # Extraire la première liste (car tolist() renvoie une liste de listes)

    # Workout
    wrkout = workout[workout['disease'] == dis]['workout'].tolist()

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin rash': 1, 'nodal skin eruptions': 2, 'continuous sneezing': 3, 'shivering': 4, 'chills': 5, 'joint pain': 6, 'stomach pain': 7, 'acidity': 8, 'ulcers on tongue': 9, 'muscle wasting': 10, 'vomiting': 11, 'burning micturition': 12, 'spotting urination': 13, 'fatigue': 14, 'weight gain': 15, 'anxiety': 16, 'cold hands and feets': 17, 'mood swings': 18, 'weight loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches in throat': 22, 'irregular sugar level': 23, 'cough': 24, 'high fever': 25, 'sunken eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish skin': 32, 'dark urine': 33, 'nausea': 34, 'loss of appetite': 35, 'pain behind the eyes': 36, 'back pain': 37, 'constipation': 38, 'abdominal pain': 39, 'diarrhoea': 40, 'mild fever': 41, 'yellow urine': 42, 'yellowing of eyes': 43, 'acute liver failure': 44, 'fluid overload': 45, 'swelling of stomach': 46, 'swelled lymph nodes': 47, 'malaise': 48, 'blurred and distorted vision': 49, 'phlegm': 50, 'throat irritation': 51, 'redness of eyes': 52, 'sinus pressure': 53, 'runny nose': 54, 'congestion': 55, 'chest pain': 56, 'weakness in limbs': 57, 'fast heart rate': 58, 'pain during bowel movements': 59, 'pain in anal region': 60, 'bloody stool': 61, 'irritation in anus': 62, 'neck pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen legs': 68, 'swollen blood vessels': 69, 'puffy face and eyes': 70, 'enlarged thyroid': 71, 'brittle nails': 72, 'swollen extremeties': 73, 'excessive hunger': 74, 'extra marital contacts': 75, 'drying and tingling lips': 76, 'slurred speech': 77, 'knee pain': 78, 'hip joint pain': 79, 'muscle weakness': 80, 'stiff neck': 81, 'swelling joints': 82, 'movement stiffness': 83, 'spinning movements': 84, 'loss of balance': 85, 'unsteadiness': 86, 'weakness of one body side': 87, 'loss of smell': 88, 'bladder discomfort': 89, 'foul smell of urine': 90, 'continuous feel of urine': 91, 'passage of gases': 92, 'internal itching': 93, 'toxic look (typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle pain': 97, 'altered sensorium': 98, 'red spots over body': 99, 'belly pain': 100, 'abnormal menstruation': 101, 'dischromic patches': 102, 'watering from eyes': 103, 'increased appetite': 104, 'polyuria': 105, 'family history': 106, 'mucoid sputum': 107, 'rusty sputum': 108, 'lack of concentration': 109, 'visual disturbances': 110, 'receiving blood transfusion': 111, 'receiving unsterile injections': 112, 'coma': 113, 'stomach bleeding': 114, 'distention of abdomen': 115, 'history of alcohol consumption': 116, 'fluid overload.1': 117, 'blood in sputum': 118, 'prominent veins on calf': 119, 'palpitations': 120, 'painful walking': 121, 'pus filled pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin peeling': 125, 'silver like dusting': 126, 'small dents in nails': 127, 'inflammatory nails': 128, 'blister': 129, 'red sore around nose': 130, 'yellow crust ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Fonction de prédiction du modèle
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    valid_symptoms = []  # Pour stocker les symptômes valides

    for item in patient_symptoms:
        # Vérifier si le symptôme existe dans le dictionnaire
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
            valid_symptoms.append(item)
        else:
            print(f"Symptôme non reconnu ignoré : {item}")

    # Si aucun symptôme valide n'est trouvé, retourner None
    if not valid_symptoms:
        return None

    # Prédire la maladie
    return diseases_list[svc.predict([input_vector])[0]]

# Injecter la liste des symptômes dans le template
@app.context_processor
def inject_all_symptoms():
    return {'all_symptoms': list(symptoms_dict.keys())}

# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            # Traitement des symptômes
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

            # Prédiction de la maladie
            predicted_disease = get_predicted_value(user_symptoms)

            # Si aucun symptôme valide n'est trouvé
            if predicted_disease is None:
                message = "No valid symptoms have been entered. Please enter valid symptoms."
                return render_template('index.html', message=message)

            # Récupération des détails (description, précautions, médicaments, régime, exercices)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            # Transformation des précautions en liste simple
            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            # Passer les données au template
            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')




@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

# ============================================================
# Point d'entrée de l'application
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
