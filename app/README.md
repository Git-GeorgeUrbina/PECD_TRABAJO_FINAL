# TRABAJO FINAL

## INTEGRANTES:
## Sarai Cisneros
## Noemi Guerra
## George Urbina
## Henry Cardenas
## Juan Auculli

## Descripción
El objetivo del proyecto es desarrollar y evaluar un modelo de Machine Learning y Deep Learning capaz de predecir el riesgo de diabetes a partir de variables clínicas, comparando distintos algoritmos, permitiendo comprender la influencia de cada variable en la predicción y facilitando su uso como herramienta de apoyo a la toma de decisiones clínicas.

## Requisitos
```bash
pip install -r requirements.txt
```
## Pasos
- Ingresar a la carpeta notebook
- Ubicar el archivo Control_4.ipynb y ejecutar todo
- Se generán los archivos model_joblib.joblib y mode_metadata.json dentro de la captera app/models
- Para realizar el deply:


## Para realizar e Deploy
```bash
conda activate ml_pro
cd app
streamlit run app.py
