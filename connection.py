import pandas as pd
import chardet
from pathlib import Path
import os

class ConectarBase:
    def __init__(self, csvDoc):
        self.csvDoc = csvDoc

    def traer(self):
        try:
            encod = self.detectar_encoding()
            base = pd.read_csv(self.csvDoc, sep=';', encoding=encod, low_memory=False)
            base = base.dropna()  # Elimina filas con valores faltantes
            return pd.DataFrame(base)
        
        except Exception as e:
            print(f"Error al cargar la base: {e}")
            return pd.DataFrame()  # Retorna un DataFrame vacío si algo falla

    def detectar_encoding(self):
        with open(self.csvDoc, 'rb') as f:
            resultado = chardet.detect(f.read(100000)) 
        return resultado['encoding']
    
    #Imprime las primeras filas de cada DataFrame en la carpeta "bases"
    def imprimirHead(self):
        df = self.traer()
        if df.empty:
            print("El DataFrame está vacío o hubo un error al cargar los datos.")
        else:
            print(df.head())  # Muestra las primeras filas del DataFrame


if __name__ == "__main__":
    carpeta = Path(os.path.dirname(os.path.abspath(__file__)))
    for archivo_csv in carpeta.glob('*.csv'):
        baseName = ConectarBase(archivo_csv)
        print(f"Archivo: {archivo_csv.name}:")
        baseName.imprimirHead()  # Ya no es necesario hacer print(baseName.imprimir())
        print()  # Salto de línea para mayor legibilidad
