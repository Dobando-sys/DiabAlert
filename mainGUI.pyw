
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from PIL import Image, ImageTk

class DiabalertGUI:
    def __init__(self, master):
        self.master = master
        master.title("Diagnóstico de Diabetes")
        master.geometry("530x500")

        self.pipeline = joblib.load("modelo_prediccion_diabetes.pkl")
        self.columnas = joblib.load("columnas_modelo.pkl")

        self.campos = {
            'edad': tk.IntVar(),
            'sexo': tk.StringVar(),
            'estrato': tk.IntVar(),
            'imc': tk.DoubleVar(),
            'glucosa_ayunas': tk.DoubleVar(),
            'presion_arterial': tk.DoubleVar(),
            'poliuria': tk.IntVar(),
            'polidipsia': tk.IntVar(),
            'perdida_peso': tk.IntVar(),
            'fatiga': tk.IntVar(),
            'vision_borrosa': tk.IntVar(),
            'curacion_lenta': tk.IntVar(),
            'antecedentes_familiares': tk.IntVar()
        }

        self.crear_formulario()
    
    def crear_formulario(self):
        frameLogo = tk.Frame(self.master)
        frameLogo.grid(row=0, column=0, columnspan=3, pady=4)
        imagen = Image.open("logo.jpg")

        # Obtener tamaño original y calcular nuevo tamaño (10%)
        ancho, alto = imagen.size
        nuevo_tamano = (int(ancho * 0.2), int(alto * 0.2))

        # Redimensionar usando LANCZOS (antes ANTIALIAS)
        imagen_reescalada = imagen.resize(nuevo_tamano, Image.Resampling.LANCZOS)

        # Convertir a formato Tkinter
        self.logo = ImageTk.PhotoImage(imagen_reescalada)

        # Mostrar en Label
        lbLogo = tk.Label(frameLogo, image=self.logo)
        lbLogo.grid(row=0, column=0, padx=10, pady=4)


        
        frame = tk.Frame(self.master)
        frame.grid(row=1, column=0, columnspan=3, padx=10, pady=4)
        
        self.frameboton = tk.Frame(self.master)
        self.frameboton.grid(row=2, column=0, columnspan=3, pady=4)

        campos_lista = list(self.campos.items())
        total = len(campos_lista)
        columnas = 3
        filas = 5

        for idx, (campo, var) in enumerate(campos_lista):
            col = idx % columnas
            row = idx // columnas

            label = tk.Label(frame, text=campo.replace("_", " ").capitalize() + ":")
            label.grid(row=row * 2, column=col, padx=10, pady=2, sticky=tk.W)

            if campo == 'sexo':
                widget = ttk.Combobox(frame, textvariable=var, values=["M", "F"], state="readonly")
            elif campo in ['poliuria', 'polidipsia', 'perdida_peso', 'fatiga',
                           'vision_borrosa', 'curacion_lenta', 'antecedentes_familiares']:
                widget = ttk.Combobox(frame, textvariable=var, values=[0, 1], state="readonly")
            else:
                widget = tk.Entry(frame, textvariable=var)

            widget.grid(row=row * 2 + 1, column=col, padx=10, pady=2)

        tk.Button(self.frameboton, text="CALCULAR", command=self.predecir,width=20, font=('Arial',10,'bold')).pack(pady=5)

    def predecir(self):
        try:
            datos = {
                k: str(v.get()) if k in [
                    'sexo', 'estrato', 'poliuria', 'polidipsia',
                    'perdida_peso', 'fatiga', 'vision_borrosa',
                    'curacion_lenta', 'antecedentes_familiares'
                ] else v.get()
                for k, v in self.campos.items()
            }

            df = pd.DataFrame([datos])

            # Agregar rango_edad (en string como lo espera el modelo)
            edad = int(datos['edad'])
            rangos = pd.cut([edad], bins=[17, 30, 40, 50, 60, 70, 86],
                            labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71-85'])
            df['rango_edad'] = str(rangos[0])

            # Asegurar que las columnas estén en el orden correcto
            df = df[self.columnas]

            prob = self.pipeline.predict_proba(df)[0, 1]

            if prob < 0.2:
                nivel = "BAJO"
                recomendaciones = [
                    "Mantener hábitos saludables",
                    "Realizar chequeos médicos regulares",
                    "Mantener una dieta balanceada"
                ]
            elif prob < 0.5:
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

            mensaje = f"Probabilidad de diabetes: {round(prob * 100, 2)}%\nNivel de riesgo: {nivel}\n\nRecomendaciones:\n"
            mensaje += "\n".join(f"- {r}" for r in recomendaciones)
            messagebox.showinfo("Resultado", mensaje)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiabalertGUI(root)
    root.mainloop()
