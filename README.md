# Clasificación de Alzheimer con TensorFlow

Modelo de red neuronal para clasificar el estado de demencia de pacientes usando el dataset OASIS Longitudinal.

## Descripción

El modelo toma 8 características clínicas y neurológicas de un paciente y predice si pertenece a una de tres categorías:

| Clase | Etiqueta | Descripción |
|-------|----------|-------------|
| 0 | Converted | Paciente que desarrolló demencia durante el estudio |
| 1 | Demented | Paciente con Alzheimer diagnosticado |
| 2 | Nondemented | Paciente sin demencia |

## Dataset

Se usa el archivo `oasis_longitudinal.csv` del proyecto [OASIS Brains](https://www.oasis-brains.org/), que contiene 373 registros de resonancias magnéticas longitudinales.

**Features utilizadas:**

| Feature | Descripción |
|---------|-------------|
| `Age` | Edad del paciente |
| `EDUC` | Años de educación |
| `SES` | Nivel socioeconómico |
| `MMSE` | Mini-Mental State Examination |
| `eTIV` | Volumen intracraneal total estimado |
| `nWBV` | Volumen cerebral total normalizado |
| `ASF` | Factor de escala Atlas |
| `Sexo` | Sexo del paciente (M=1, F=0) |

## Estructura del proyecto

```
├── alzheimer_modelo.py     # Código principal
├── oasis_longitudinal.csv  # Dataset (no incluido en el repo)
├── modelo_alzheimer.keras  # Modelo entrenado (se genera al correr)
└── README.md
```

## Instalación

```bash
pip install tensorflow numpy pandas scikit-learn
```

## Uso

Coloca `oasis_longitudinal.csv` en la misma carpeta que el script y ejecuta:

```bash
python alzheimer_modelo.py
```

El script realiza automáticamente los siguientes pasos:

1. Carga y limpia el dataset
2. Normaliza las features (z-score)
3. Divide en entrenamiento/validación (80/20)
4. Construye y entrena la red neuronal
5. Evalúa el accuracy en validación
6. Guarda el modelo en `modelo_alzheimer.keras`

## Arquitectura del modelo

```
Input (8)  →  Dense(64, relu)  →  Dropout(0.3)  →  Dense(32, relu)  →  Dense(3, softmax)
```

- **Optimizador:** Adam  
- **Loss:** Sparse Categorical Crossentropy  
- **Épocas:** 50  
- **Batch size:** 16  

## Funciones principales

- `cargar_dataset()` — Lee el CSV, limpia nulos, codifica variables y retorna X, y
- `construir_modelo(num_classes)` — Define y compila la red neuronal
- `entrenar_modelo(...)` — Entrena el modelo con datos de entrenamiento y validación
- `guardar_modelo(...)` — Guarda el modelo entrenado en disco
