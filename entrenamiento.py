import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


class Entrenador:
    def __init__(self):
        self.df = self.cargar_datos_diabetes_colombia()

        self.df['rango_edad'] = pd.cut(self.df['edad'], bins=[17, 30, 40, 50, 60, 70, 86],
                                       labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71-85'])

        categorical_columns = ['sexo', 'estrato', 'rango_edad', 'poliuria', 'polidipsia',
                               'perdida_peso', 'fatiga', 'vision_borrosa', 'curacion_lenta',
                               'antecedentes_familiares']

        for col in categorical_columns:
            if col == "rango_edad":
                self.df[col] = self.df[col].astype(str).fillna('Desconocido')
            else:
                self.df[col] = self.df[col].fillna('Desconocido').astype(str)

        self.X = self.df.drop('diabetes', axis=1)
        self.y = self.df['diabetes']

        self.cat_features = categorical_columns
        self.num_features = ['edad', 'imc', 'glucosa_ayunas', 'presion_arterial']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)
            ])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        self.pipeline.fit(self.X_train, self.y_train)

        joblib.dump(self.pipeline, 'modelo_prediccion_diabetes.pkl')
        joblib.dump(self.X.columns.tolist(), 'columnas_modelo.pkl')

    def cargar_datos_diabetes_colombia(self):
        np.random.seed(42)
        n_samples = 1000

        sexo = np.random.choice(['M', 'F'], size=n_samples)
        edad = np.clip(np.random.normal(45, 15, n_samples).astype(int), 18, 85)
        estrato = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples)

        imc = np.clip(np.random.normal(26, 4, n_samples), 16, 45)
        glucosa = np.clip(np.random.normal(95, 20, n_samples) + (7 - estrato) * 2, 70, 300)
        presion = np.clip(120 + (edad - 40) * 0.2 + np.random.normal(0, 15, n_samples), 90, 200)

        poliuria = (glucosa > 160) & (np.random.rand(n_samples) < 0.8)
        polidipsia = (poliuria | (glucosa > 150)) & (np.random.rand(n_samples) < 0.7)
        perdida_peso = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
        fatiga = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        vision_borrosa = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
        curacion_lenta = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        antecedentes_familiares = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

        diabetes = []
        for i in range(n_samples):
            score = 0
            if glucosa[i] > 125: score += 4
            elif glucosa[i] > 100: score += 2
            if imc[i] > 30: score += 2
            elif imc[i] > 25: score += 1
            if edad[i] > 45: score += 1
            if antecedentes_familiares[i]: score += 2
            if poliuria[i]: score += 1
            if polidipsia[i]: score += 1
            if perdida_peso[i]: score += 1
            if vision_borrosa[i]: score += 1
            if curacion_lenta[i]: score += 1
            if estrato[i] <= 2: score += 0.5
            prob = 1 / (1 + np.exp(-0.5 * (score - 6)))
            diabetes.append(int(np.random.rand() < prob))

        df = pd.DataFrame({
            'sexo': sexo,
            'edad': edad,
            'estrato': estrato,
            'imc': imc,
            'glucosa_ayunas': glucosa,
            'presion_arterial': presion,
            'poliuria': poliuria.astype(int),
            'polidipsia': polidipsia.astype(int),
            'perdida_peso': perdida_peso,
            'fatiga': fatiga,
            'vision_borrosa': vision_borrosa,
            'curacion_lenta': curacion_lenta,
            'antecedentes_familiares': antecedentes_familiares,
            'diabetes': diabetes
        })

        return df


class Diabalert:
    def __init__(self):
        self.pipeline = joblib.load('modelo_prediccion_diabetes.pkl')
        self.columnas = joblib.load('columnas_modelo.pkl')

    def predecir_diabetes(self, datos_paciente):
        df = pd.DataFrame([datos_paciente], columns=self.columnas)

        # Agregar rango_edad
        df['rango_edad'] = pd.cut(df['edad'], bins=[17, 30, 40, 50, 60, 70, 86],
                                  labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71-85'])

        # Convertir a string los valores categóricos
        for col in ['sexo', 'estrato', 'rango_edad', 'poliuria', 'polidipsia',
                    'perdida_peso', 'fatiga', 'vision_borrosa', 'curacion_lenta',
                    'antecedentes_familiares']:
            if col in df.columns:
                df[col] = df[col].astype(str)

        proba = self.pipeline.predict_proba(df)[0, 1]

        if proba < 0.2:
            nivel = "BAJO"
            recomendaciones = [
                "Mantener hábitos saludables",
                "Realizar chequeos médicos regulares",
                "Mantener una dieta balanceada"
            ]
        elif proba < 0.5:
            nivel = "MODERADO"
            recomendaciones = [
                "Consultar con un médico para evaluación",
                "Considerar pruebas de glucosa en ayunas",
                "Revisar hábitos alimenticios",
                "Aumentar actividad física"
            ]
        else:
            nivel = "ALTO"
            recomendaciones = [
                "Consultar con un médico urgentemente",
                "Realizar pruebas de glucosa en ayunas y HbA1c",
                "Control de factores de riesgo",
                "Posible inicio de tratamiento preventivo"
            ]

        return {
            'probabilidad': round(proba * 100, 2),
            'nivel_riesgo': nivel,
            'recomendaciones': recomendaciones
        }

    def sistema_diagnostico_diabetes(self):
        print("Sistema de diagnóstico de diabetes")
        try:
            datos = {
                'edad': int(input("Edad: ")),
                'sexo': input("Sexo (M/F): ").upper(),
                'estrato': int(input("Estrato (1-6): ")),
                'imc': float(input("IMC: ")),
                'glucosa_ayunas': float(input("Glucosa en ayunas: ")),
                'presion_arterial': float(input("Presión arterial: ")),
                'poliuria': int(input("¿Poliuria? (1/0): ")),
                'polidipsia': int(input("¿Polidipsia? (1/0): ")),
                'perdida_peso': int(input("¿Pérdida de peso? (1/0): ")),
                'fatiga': int(input("¿Fatiga? (1/0): ")),
                'vision_borrosa': int(input("¿Visión borrosa? (1/0): ")),
                'curacion_lenta': int(input("¿Curación lenta? (1/0): ")),
                'antecedentes_familiares': int(input("¿Antecedentes familiares? (1/0): "))
            }

            resultado = self.predecir_diabetes(datos)
            print(f"\nProbabilidad de diabetes: {resultado['probabilidad']}%")
            print(f"Nivel de riesgo: {resultado['nivel_riesgo']}")
            print("Recomendaciones:")
            for r in resultado['recomendaciones']:
                print(f"- {r}")
        except Exception as e:
            print("Error en la entrada de datos:", e)


# Si deseas entrenar manualmente, descomenta la línea:
Entrenador()
