import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 

def convertir_rango_edad(rango):
    # Separa los valores del rango por el guion
    limites = rango.split('-')
    # Calcula el promedio de esos dos valores
    return (int(limites[0]) + int(limites[1])) / 2

# Función para cargar los datos 
# Nota: En un caso real, esta función cargaría datos reales de Colombia 

def cargar_datos_diabetes_colombia(): 
    """ 
    En un escenario real, esta función cargaría datos específicos de Colombia. 
    Por ahora, generamos datos sintéticos que simulan una población colombiana. 
    """ 
    # Características simuladas 
    np.random.seed(42) 
    n_samples = 1000 

    # Datos demográficos 
    sexo = np.random.choice(['M', 'F'], size=n_samples, p=[0.48, 0.52]) 
    edad = np.random.normal(45, 15, n_samples).astype(int) 
    edad = np.clip(edad, 18, 85)  # Limitar edad entre 18 y 85 años 
    estrato = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.20, 0.25, 0.30, 0.15, 0.07, 0.03]) 

    # Factores clínicos - simulados con correlaciones realistas 
    # Índice de masa corporal (IMC) 
    imc = np.random.normal(26, 4, n_samples) 
    imc = np.clip(imc, 16, 45) 

    # Glucosa en ayunas (mg/dL) 
    # Valores normales: 70-100, prediabetes: 100-125, diabetes: >126 
    base_glucosa = np.random.normal(95, 20, n_samples) 

    # Ajuste por estrato (correlación negativa) 
    estrato_factor = (7 - estrato) * 2 
    glucosa = base_glucosa + estrato_factor 
    glucosa = np.clip(glucosa, 70, 300) 
    
    # Presión arterial sistólica (mmHg) 
    presion = np.random.normal(120, 15, n_samples) 
    
    # La presión aumenta con la edad 
    presion = presion + (edad - 40) * 0.2 
    presion = np.clip(presion, 90, 200) 

    # Poliuria (orinar frecuentemente) - más común en diabéticos 
    poliuria = np.random.choice([0, 1], size=n_samples) 
    
    # Ajustar poliuria basado en niveles de glucosa 
    for i in range(n_samples): 
        if glucosa[i] > 160: 
            if np.random.random() < 0.8:  # 80% de probabilidad si glucosa alta 
                poliuria[i] = 1 

    # Polidipsia (sed excesiva) - más común en diabéticos 
    polidipsia = np.random.choice([0, 1], size=n_samples) 

    # Correlacionar con poliuria y glucosa 
    for i in range(n_samples): 
        if poliuria[i] == 1 or glucosa[i] > 150: 
            if np.random.random() < 0.7: 
                polidipsia[i] = 1 

    # Pérdida de peso inexplicable 
    perdida_peso = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]) 

    # Fatiga 
    fatiga = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) 

    # Visión borrosa 
    vision_borrosa = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]) 

    # Heridas de curación lenta 
    curación_lenta = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]) 

    # Antecedentes familiares 
    antecedentes_familiares = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) 

    # Diabetes (resultado) - basado en factores de riesgo 
    diabetes = np.zeros(n_samples, dtype=int) 

    for i in range(n_samples): 
        # Calcular puntaje de riesgo basado en múltiples factores 
        score = 0 
        if glucosa[i] > 125: 
            score += 4 
        elif glucosa[i] > 100: 
            score += 2 

        if imc[i] > 30: 
            score += 2 
        elif imc[i] > 25: 
            score += 1 

        if edad[i] > 45: 
            score += 1 

        if antecedentes_familiares[i] == 1: 
            score += 2 

        if poliuria[i] == 1: 
            score += 1 

        if polidipsia[i] == 1: 
            score += 1 

        if perdida_peso[i] == 1: 
            score += 1 

        if vision_borrosa[i] == 1: 
            score += 1 

        if curación_lenta[i] == 1: 
            score += 1 

        # Estrato afecta por acceso a servicios de salud (factor social) 
        if estrato[i] <= 2: 
            score += 0.5 

        # Determinar diagnóstico según puntaje 
        prob_diabetes = 1 / (1 + np.exp(-0.5 * (score - 6)))  # Función sigmoide 
        if np.random.random() < prob_diabetes: 
            diabetes[i] = 1 

    # Crear dataframe 
    df = pd.DataFrame({ 
        'sexo': sexo, 
        'edad': edad, 
        'estrato': estrato, 
        'imc': imc, 
        'glucosa_ayunas': glucosa, 
        'presion_arterial': presion, 
        'poliuria': poliuria, 
        'polidipsia': polidipsia, 
        'perdida_peso': perdida_peso, 
        'fatiga': fatiga, 
        'vision_borrosa': vision_borrosa, 
        'curacion_lenta': curación_lenta, 
        'antecedentes_familiares': antecedentes_familiares, 
        'diabetes': diabetes 
    }) 

    return df 

# Cargar los datos 
df = cargar_datos_diabetes_colombia() 

# Mostrar estadísticas descriptivas 
print("Estadísticas descriptivas de la muestra:") 
print(df.describe()) 

# Mostrar distribución de diabetes por sexo, estrato y rangos de edad 
print("\nDistribución de diabetes por sexo:") 
print(pd.crosstab(df['sexo'], df['diabetes'], normalize='index') * 100) 

print("\nDistribución de diabetes por estrato:") 
print(pd.crosstab(df['estrato'], df['diabetes'], normalize='index') * 100) 

# Crear rangos de edad 
df['rango_edad'] = pd.cut(df['edad'], bins=[17, 30, 40, 50, 60, 70, 86],  
                        labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71-85']) 

print("\nDistribución de diabetes por rango de edad:") 
print(pd.crosstab(df['rango_edad'], df['diabetes'], normalize='index') * 100) 

# Análisis de factores de riesgo 
print("\nFactores más correlacionados con diabetes:") 

df['sexo'] = df['sexo'].map({'M': 1, 'F': 0})
df['edad_promedio'] = df['rango_edad'].apply(convertir_rango_edad)
df_numeric = df.select_dtypes(include=['number'])
df_numeric = df_numeric.dropna()  # Elimina filas con valores nulos
correlacion = df_numeric.corr()['diabetes'].sort_values(ascending=False)

print(correlacion) 

# Visualizaciones 
plt.figure(figsize=(10, 6)) 
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5) 
plt.title('Matriz de Correlación de Variables') 
plt.tight_layout() 

# plt.show() - Comentado para producción, pero puedes usarlo en análisis 
# Preparar datos para modelado 
X = df.drop('diabetes', axis=1) 
y = df['diabetes'] 

# Definir características categóricas y numéricas 
cat_features = ['sexo', 'estrato', 'rango_edad', 'poliuria', 'polidipsia',  
                'perdida_peso', 'fatiga', 'vision_borrosa',  
                'curacion_lenta', 'antecedentes_familiares'] 
num_features = ['edad', 'imc', 'glucosa_ayunas', 'presion_arterial'] 

# Crear preprocesadores para cada tipo de característica 
preprocessor = ColumnTransformer( 
    transformers=[ 
        ('num', StandardScaler(), num_features), 
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features) 
    ]) 

# Dividir datos en conjuntos de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

# Crear pipeline con preprocesamiento y modelo 
pipeline = Pipeline(steps=[ 
    ('preprocessor', preprocessor), 
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) 
]) 

# Entrenar el modelo 
pipeline.fit(X_train, y_train) 

# Evaluar el modelo 
y_pred = pipeline.predict(X_test) 
y_proba = pipeline.predict_proba(X_test)[:, 1] 

print("\nInforme de clasificación:") 
print(classification_report(y_test, y_pred)) 
print("\nMatriz de confusión:") 
print(confusion_matrix(y_test, y_pred)) 

# Función para predecir riesgo de diabetes en nuevos pacientes 
def predecir_diabetes(pipeline, datos_paciente): 
    """ 
    Predice la probabilidad de diabetes para un paciente. 
    Args: 
        pipeline: Pipeline entrenado 
        datos_paciente: DataFrame con datos del paciente 

    Returns: 
        Probabilidad de diabetes y diagnóstico 
    """ 

    # Verificar datos faltantes 
    columnas_faltantes = set(X.columns) - set(datos_paciente.columns) 
    if columnas_faltantes: 
        raise ValueError(f"Faltan las siguientes columnas: {columnas_faltantes}") 

    # Predecir probabilidad 
    prob_diabetes = pipeline.predict_proba(datos_paciente)[0, 1] 

    # Determinar diagnóstico según probabilidad 
    if prob_diabetes < 0.2: 
        nivel_riesgo = "BAJO" 
        recomendaciones = [ 
            "Mantener hábitos saludables", 
            "Realizar chequeos médicos regulares", 
            "Mantener una dieta balanceada" 
        ] 

    elif prob_diabetes < 0.5: 
        nivel_riesgo = "MODERADO" 
        recomendaciones = [ 
            "Consultar con un médico para evaluación", 
            "Considerar pruebas de glucosa en ayunas", 
            "Revisar hábitos alimenticios", 
            "Aumentar actividad física" 
        ] 
        
    else: 
        nivel_riesgo = "ALTO" 
        recomendaciones = [ 
            "Consultar con un médico urgentemente", 
            "Realizar pruebas de glucosa en ayunas y HbA1c", 
            "Control de factores de riesgo", 
            "Posible inicio de tratamiento preventivo" 
        ] 

    return { 
        'probabilidad': round(prob_diabetes * 100, 2), 
        'nivel_riesgo': nivel_riesgo, 
        'recomendaciones': recomendaciones 
    } 

# Ejemplo de uso 
def sistema_diagnostico_diabetes(): 
    print("\n--- SISTEMA DE DIAGNÓSTICO TEMPRANO DE DIABETES ---\n") 

    # Recopilación de datos del paciente 
    datos = {} 
    
    # Información demográfica 
    print("Información demográfica:") 
    datos['sexo'] = input("Sexo (M/F): ").upper() 
    datos['edad'] = int(input("Edad: ")) 
    datos['estrato'] = int(input("Estrato socioeconómico (1-6): ")) 
    datos['rango_edad'] = pd.cut([datos['edad']], bins=[17, 30, 40, 50, 60, 70, 86],  
                                labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71-85'])[0] 

    # Medidas clínicas 
    print("\nMedidas clínicas:") 
    datos['imc'] = float(input("Índice de Masa Corporal (IMC): ")) 
    datos['glucosa_ayunas'] = float(input("Nivel de glucosa en ayunas (mg/dL): ")) 
    datos['presion_arterial'] = float(input("Presión arterial sistólica (mmHg): ")) 
    
    # Síntomas 
    print("\nSíntomas (responda 1 para Sí, 0 para No):") 
    datos['poliuria'] = int(input("¿Orina con frecuencia? ")) 
    datos['polidipsia'] = int(input("¿Tiene sed excesiva? ")) 
    datos['perdida_peso'] = int(input("¿Ha tenido pérdida de peso inexplicable? ")) 
    datos['fatiga'] = int(input("¿Siente fatiga inusual? ")) 
    datos['vision_borrosa'] = int(input("¿Ha experimentado visión borrosa? ")) 
    datos['curacion_lenta'] = int(input("¿Sus heridas tardan en sanar? ")) 

    # Antecedentes 
    datos['antecedentes_familiares'] = int(input("\n¿Tiene familiares con diabetes? (1=Sí, 0=No): ")) 

    # Crear DataFrame con los datos del paciente 
    df_paciente = pd.DataFrame([datos]) 

    # Predecir riesgo de diabetes 
    resultado = predecir_diabetes(pipeline, df_paciente) 

    # Mostrar resultados 
    print("\n--- RESULTADO DEL DIAGNÓSTICO ---") 
    print(f"Probabilidad de diabetes: {resultado['probabilidad']}%") 
    print(f"Nivel de riesgo: {resultado['nivel_riesgo']}") 
    print("\nRecomendaciones:") 
    
    for rec in resultado['recomendaciones']: 
        print(f"- {rec}") 

    print("\nNOTA IMPORTANTE: Este es solo un diagnóstico preliminar.") 
    print("Consulte siempre con un profesional de la salud para una evaluación completa.") 

# Para usar el sistema interactivo, descomenta la siguiente línea 
# sistema_diagnostico_diabetes() 

# Ejemplo de paciente para demostración 
def demostrar_sistema(): 
    print("\n--- DEMOSTRACIÓN DEL SISTEMA ---") 

    # Paciente de ejemplo 
    paciente_demo = pd.DataFrame([{ 
        'sexo': 'F', 
        'edad': 58, 
        'estrato': 3, 
        'rango_edad': '51-60', 
        'imc': 32.1, 
        'glucosa_ayunas': 135, 
        'presion_arterial': 145, 
        'poliuria': 1, 
        'polidipsia': 1, 
        'perdida_peso': 0, 
        'fatiga': 1, 
        'vision_borrosa': 1, 
        'curacion_lenta': 1, 
        'antecedentes_familiares': 1,
        'edad_promedio': 58
    }]) 

    # Predecir riesgo 
    resultado = predecir_diabetes(pipeline, paciente_demo) 

    # Mostrar resultados 
    print("\nDatos del paciente de ejemplo:") 
    print(f"- Sexo: Femenino") 
    print(f"- Edad: 58 años") 
    print(f"- Estrato: 3") 
    print(f"- IMC: 32.1 (Obesidad grado I)") 
    print(f"- Glucosa en ayunas: 135 mg/dL (Elevada)") 
    print(f"- Presión arterial: 145 mmHg (Hipertensión etapa 1)") 
    print(f"- Síntomas: Poliuria, polidipsia, fatiga, visión borrosa, heridas de curación lenta") 
    print(f"- Antecedentes familiares: Sí") 
    
    print("\n--- RESULTADO DEL DIAGNÓSTICO ---") 
    print(f"Probabilidad de diabetes: {resultado['probabilidad']}%") 
    print(f"Nivel de riesgo: {resultado['nivel_riesgo']}") 
    print("\nRecomendaciones:") 
    
    for rec in resultado['recomendaciones']: 
        print(f"- {rec}") 

# Demostrar sistema con un paciente de ejemplo 
demostrar_sistema() 

# Para implementación en producción, se guardaría el modelo entrenado 
# import joblib 
# joblib.dump(pipeline, 'modelo_prediccion_diabetes.pkl') 
