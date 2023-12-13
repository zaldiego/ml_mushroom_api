from fastapi import FastAPI, HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import joblib



#Base de datos
DATABASE = "/home/zaldiego/Escritorio/ESTUDIO/machine_learning/dataset/mushrooms.csv"



class HongosModel:
    def __init__(self):
        #Declarar variables del modelo y del codificador
        self.model_fname = 'trained_model_mushroom.pkl'
        self.ohe_fname = 'trained_ohe_mushroom.pkl'
        self.model = None
        self.ohe = None

        #Ejecución y carga del modelo y del codificador
        try:
            self.model = joblib.load(self.model_fname)
            self.ohe = joblib.load(self.ohe_fname)
        except Exception as _:
            self.model, self.ohe = self.train_model()
            joblib.dump(self.model, self.model_fname)
            joblib.dump(self.ohe, self.ohe_fname)



    def train_model(self):

        #Importar los datos y guardarlos como un dataframe
        dataset = pd.read_csv(DATABASE, header=0)

        df = dataset[['class', 'cap_shape', 'cap_surface', 'bruises', 'odor', 'gill_attachment',
             'gill_spacing', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
             'spore_print_color', 'habitat']]
        
        #Construir un dataframe tanto para X como para Y
        X_df = df.drop(columns=['class'])
        y_df = df['class']

        #Codificar ambos dataframes
        ohe = OneHotEncoder(handle_unknown='ignore')
        X_coded = ohe.fit_transform(X_df).toarray()

        label = LabelEncoder()
        y_label = label.fit_transform(y_df)
        y_coded = pd.Series(y_label)


        # Split de los datos
        X_train, X_test, y_train, y_test = train_test_split(X_coded, y_coded, test_size=0.3, random_state=0)

        #Seleccionar el modelo y sus parametros, previamente estudiado en Google Colab
        clf = RandomForestClassifier(max_depth=10, n_estimators= 50)

        #Entrenar el modelo
        clf.fit(X_train, y_train)

        #Obtener el modelo y el codificador entrenado para ejecutar la predicción
        return clf, ohe



    def model_prediction(self, cap_shape, cap_surface, bruises, odor, gill_attachment,
                        gill_spacing, veil_type, veil_color, ring_number, ring_type,
                        spore_print_color, habitat):
        
        #Parametros de input del usuario
        data_in = [[cap_shape, cap_surface, bruises, odor, gill_attachment,
                    gill_spacing, veil_type, veil_color, ring_number, ring_type,
                    spore_print_color, habitat]]

        #Si ocurre un error en el modelo o en el codificador el programa devuelve un mensaje de error
        if self.model is None or self.ohe is None:
            raise HTTPException(status_code=500, detail="El modelo no está entrenado. Primero, debe entrenar el modelo.")

        # Transformar los datos de entrada utilizando el OneHotEncoder entrenado
        data_in_encoded = self.ohe.transform(data_in).toarray()

        # Realizar la predicción directamente con la matriz transformada
        prediction = self.model.predict(data_in_encoded)
        probability = self.model.predict_proba(data_in_encoded).max()

        #Devuelve los resultados de clasificación y de probabilidad de pertenencia a dicha clase
        return int(prediction[0]), float(probability)



'''
Este codigo es un ejemplo de estructura básica de una API de Machine Learning hecha con FastAPI, 
para evitar potenciales errores futuros en el codigo de la API. Tanto si el modelo lo hago yo 
como si lo hace un asociado a Neural Batch primero se debe pasar por Google Colab o Jupyter Notebook 
para realmente poner a prueba el modelo y todos los datos transformados e impresos en sus variables. 

De esa forma puedo asegurarme de tener toda la información correcta para poder empezar a hacer
el desarrollo de la API de manera segura con todas las variables. 
'''

        