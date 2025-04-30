import pandas as pd
import chardet
import os

class ConectarBase:
    def __init__(self, csvDoc):
        self.csvDoc = csvDoc

    def traer(self):
        try:
            encod = self.detectar_encoding()
            base = pd.read_csv(self.csvDoc, sep=';', encoding=encod, low_memory=False)
            base = base.dropna()
            
            return pd.DataFrame(base)
        
        except Exception as e:
            print(f"Error al cargar la base: {e}")
            return pd.DataFrame()  # Retorna un DataFrame vacío si algo falla

    
    def detectar_encoding(self):
        with open(self.csvDoc, 'rb') as f:
            resultado = chardet.detect(f.read(100000))  # Lee los primeros 100KB
        
        return resultado['encoding']
    
    def imprimir(self):
        df = self.traer()
        print(df.head())

# Definición de las bases de datos
# Se definen las bases de datos a utilizar, junto con sus rutas locales (Bases en csv) y URLs de consulta en linea.
bases ={"Enfermedades Cronicas en Bucaramanga": ["C:/Users/esteban.obando/OneDrive - SEMILLAS ELITE DE PALMA PARA LAS AMÉRICAS S.A.S/Documentos/GitHub/DiabAlert/bases/Enfermedades Cronicas Bucaramanga.csv",
                                                "https://www.datos.gov.co/Salud-y-Protecci-n-Social/39-Enfermedades-cr-nicas-en-el-municipio-de-Bucara/4iz7-suhz/about_data"],
        
        "Muertes Prematuras por Diabetes en Bogota":["C:/Users/esteban.obando/OneDrive - SEMILLAS ELITE DE PALMA PARA LAS AMÉRICAS S.A.S/Documentos/GitHub/DiabAlert/bases/Muertes Prematuras por Diabetes.csv",
                                                    "https://datosabiertos.bogota.gov.co/dataset/tasa-de-mortalidad-por-diabetes-mellitus-bogota-d-c/resource/43d6df80-8661-4b6d-b29a-f1e5f17e59d5?inner_span=True"],
        
        "Diabetes Mellitus no insulinodependiente":["C:/Users/esteban.obando/OneDrive - SEMILLAS ELITE DE PALMA PARA LAS AMÉRICAS S.A.S/Documentos/GitHub/DiabAlert/bases/Diabetes Mellitus no insulinodependiente.csv",
                                                "https://www.datos.gov.co/Salud-y-Protecci-n-Social/DIABETES-MELLITUS-NO-INSULINODEPENDIENTE/nf9g-8bes/about_data"]}

base = ConectarBase(bases["Diabetes Mellitus no insulinodependiente"][0])
base.imprimir()