import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from PIL import Image, ImageTk
import os
import sys

class DiabalertGUI:
    def resource_path(self, relative_path):
        """Obtiene la ruta absoluta del recurso, compatible con PyInstaller."""
        try:
            base_path = sys._MEIPASS  # Cuando está empaquetado con PyInstaller
        except AttributeError:
            base_path = os.path.abspath(".")  # Cuando se ejecuta normalmente
        return os.path.join(base_path, relative_path)
    
    def __init__(self, master):
        self.master = master
        master.title("Diabalert - diagnóstico inteligente")
        master.geometry("475x500")
        master.resizable(False, False)
        master.configure(bg="#FFFFFF")
        icon_path = self.resource_path("C:/Users/esteban.obando/OneDrive - SEMILLAS ELITE DE PALMA PARA LAS AMÉRICAS S.A.S/Documentos/GitHub/DiabAlert/COMPI/icono.ico")

        # Verifica que el archivo existe y es accesible
        if os.path.exists(icon_path):
            try:
                master.iconbitmap(icon_path)
            except tk.TclError:
                print("No se pudo cargar el ícono. Continuando sin ícono.")
        else:
            print("Ícono no encontrado. Continuando sin ícono.")


        
        self.TamCampos = 18

        self.pipeline = joblib.load(self.resource_path("C:/Users/esteban.obando/OneDrive - SEMILLAS ELITE DE PALMA PARA LAS AMÉRICAS S.A.S/Documentos/GitHub/DiabAlert/COMPI/modelo_prediccion_diabetes.pkl"))
        self.columnas = joblib.load(self.resource_path("C:/Users/esteban.obando/OneDrive - SEMILLAS ELITE DE PALMA PARA LAS AMÉRICAS S.A.S/Documentos/GitHub/DiabAlert/COMPI/columnas_modelo.pkl"))

        self.campos = {
            'edad': tk.IntVar(),
            'sexo': tk.StringVar(),
            'estrato': tk.IntVar(),
            'imc': tk.DoubleVar(),
            'glucosa_ayunas': tk.DoubleVar(),
            'presion_arterial': tk.DoubleVar(),
            'poliuria': tk.StringVar(),
            'polidipsia': tk.StringVar(),
            'perdida_peso': tk.StringVar(),
            'fatiga': tk.StringVar(),
            'vision_borrosa': tk.StringVar(),
            'curacion_lenta': tk.StringVar(),
            'antecedentes_familiares': tk.StringVar()
        }

        self.crear_formulario()
    
    def crear_formulario(self):
        frameLogo = tk.Frame(self.master,bg="#FFFFFF")
        frameLogo.grid(row=0, column=0, columnspan=3, pady=1)
        imagen = Image.open(self.resource_path("C:/Users/esteban.obando/OneDrive - SEMILLAS ELITE DE PALMA PARA LAS AMÉRICAS S.A.S/Documentos/GitHub/DiabAlert/COMPI/logo.png"))

        # Obtener tamaño original y calcular nuevo tamaño (10%)
        ancho, alto = imagen.size
        nuevo_tamano = (int(ancho * 0.55), int(alto * 0.55))

        # Redimensionar usando LANCZOS (antes ANTIALIAS)
        imagen_reescalada = imagen.resize(nuevo_tamano, Image.Resampling.LANCZOS)

        # Convertir a formato Tkinter
        self.logo = ImageTk.PhotoImage(imagen_reescalada)

        # Mostrar en Label
        lbLogo = tk.Label(frameLogo, image=self.logo,bg="#FFFFFF")
        lbLogo.grid(row=0, column=0, padx=10, pady=1)
        
        lbdescripcion = tk.Label(frameLogo, text="\nSistema de Diagnóstico de Diabetes", font=('Arial', 12, 'bold'),bg="#FFFFFF")
        lbdescripcion.grid(row=1, column=0, padx=10, pady=1)
        
        lbltexAdicional = tk.Label(frameLogo, text="Ingrese los datos a continuación para conocer su predisposición a padecer diabetes", font=('Arial', 8),bg="#FFFFFF")
        lbltexAdicional.grid(row=2, column=0, padx=10, pady=1)


        
        frame = tk.Frame(self.master, border=2, relief=tk.RIDGE,bg="#e5f0f2")
        frame.grid(row=1, column=0, columnspan=3, padx=10, pady=1)
        
        self.frameboton = tk.Frame(self.master,bg="#FFFFFF")
        self.frameboton.grid(row=2, column=0, columnspan=3, pady=3)

        campos_lista = list(self.campos.items())
        total = len(campos_lista)
        columnas = 3
        filas = 5

        for idx, (campo, var) in enumerate(campos_lista):
            col = idx % columnas
            row = idx // columnas

            label = tk.Label(frame, text=campo.replace("_", " ").capitalize() + ":",width=self.TamCampos,bg="#e5f0f2")
            label.grid(row=row * 2, column=col, padx=10, pady=2, sticky=tk.W)

            if campo == 'sexo':
                widget = ttk.Combobox(frame, textvariable=var, values=["Hombre", "Mujer"], state="readonly", width=self.TamCampos)
                widget.set("Hombre")
            elif campo in ['poliuria', 'polidipsia', 'perdida_peso', 'fatiga',
                           'vision_borrosa', 'curacion_lenta', 'antecedentes_familiares']:
                widget = ttk.Combobox(frame, textvariable=var, values=["No", "Sí"], state="readonly", width=self.TamCampos)
                widget.set("No")
            elif campo == 'edad':
                widget = tk.Spinbox(frame, textvariable=var, from_=18, to=85, width=self.TamCampos)
            elif campo == 'estrato':
                widget = tk.Spinbox(frame, textvariable=var, from_=1, to=6, width=self.TamCampos)
            else:
                widget = tk.Entry(frame, textvariable=var,width=self.TamCampos)

            widget.grid(row=row * 2 + 1, column=col, padx=10, pady=2)

        tk.Button(self.frameboton, text="CALCULAR", command=self.predecir,width=20, font=('Arial',10,'bold'),bg="#166371", foreground="#FFFFFF").pack(pady=2)

    def predecir(self):
        try:
            datos = {}
            for k, v in self.campos.items():
                valor = v.get()
                if k in ['poliuria', 'polidipsia', 'perdida_peso', 'fatiga',
                        'vision_borrosa', 'curacion_lenta', 'antecedentes_familiares']:
                    datos[k] = 1 if str(valor).lower() == "sí" else 0  # entero, no string

                elif k == 'sexo':
                    if str(valor).lower() == "mujer":
                        datos[k]  = "F"
                    elif str(valor).lower() == "hombre":
                        datos[k] = "M"
                elif k in ['edad', 'estrato']:
                    datos[k] = int(valor)
                elif k in ['imc', 'glucosa_ayunas', 'presion_arterial']:
                    datos[k] = float(valor)





            edad = datos.get('edad', None)  # Obtenemos 'edad'
            if not edad or int(edad) < 18 or int(edad) > 85:
                messagebox.showerror("Error", "La edad debe estar entre 18 y 85 años para que el modelo funcione correctamente.")
                return
            
            estrato = datos.get('estrato', None)  # Obtenemos 'edad'
            if not estrato or int(estrato) < 1 or int(estrato) > 6:
                messagebox.showerror("Error", "Por favor ingrese un estrato valido entre 1 y 6.")
                return
            
            df = pd.DataFrame([datos])
            print("\nValores faltantes:")
            print(df.isnull().sum())

            
            # Agregar rango_edad (en string como lo espera el modelo)
            
            rangos = pd.cut([int(edad)], bins=[17, 30, 40, 50, 60, 70, 86],
                            labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71-85'])
            df['rango_edad'] = rangos[0] if not pd.isnull(rangos[0]) else 'Desconocido'
            df['rango_edad'] = df['rango_edad'].astype(str)


            # Asegurar que las columnas estén en el orden correcto
            # Asegurar que los campos categóricos sean string
            categoricas = ['sexo', 'estrato', 'rango_edad', 'poliuria', 'polidipsia',
                        'perdida_peso', 'fatiga', 'vision_borrosa',
                        'curacion_lenta', 'antecedentes_familiares']

            for col in categoricas:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            df = df[self.columnas]



            print("Tipos de datos en df antes de predecir:")
            print(df.dtypes)
            print("\nValores en df:")
            print(df)
            print("Columnas esperadas por el modelo:")
            print(self.columnas)

            print("\nColumnas del DataFrame actual:")
            print(df.columns.tolist())
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
