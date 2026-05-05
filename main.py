import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def cargar_dataset():
    df = pd.read_csv('Alzheimer’s_Data_Set\oasis_longitudinal.csv')

    # Eliminar filas con valores nulos en columnas clave
    df = df.dropna(subset=['SES', 'MMSE'])

    # Codificar variables categóricas
    df['Sexo'] = (df['M/F'] == 'M').astype(np.float32)  # M=1, F=0

    # Seleccionar features numéricas relevantes
    age       = df['Age'].astype(np.float32).to_numpy()
    educ      = df['EDUC'].astype(np.float32).to_numpy()
    ses       = df['SES'].astype(np.float32).to_numpy()
    mmse      = df['MMSE'].astype(np.float32).to_numpy()
    etiv      = df['eTIV'].astype(np.float32).to_numpy()
    nwbv      = df['nWBV'].astype(np.float32).to_numpy()
    asf       = df['ASF'].astype(np.float32).to_numpy()
    sexo      = df['Sexo'].to_numpy()

    print("=== Estadísticas del Dataset ===")
    print(f"Total de registros (sin nulos): {len(df)}")
    print(f"Media Age:  {np.mean(age):.2f}")
    print(f"Media EDUC: {np.mean(educ):.2f}")
    print(f"Media SES:  {np.mean(ses):.2f}")
    print(f"Media MMSE: {np.mean(mmse):.2f}")
    print(f"Media eTIV: {np.mean(etiv):.2f}")
    print(f"Media nWBV: {np.mean(nwbv):.4f}")
    print(f"Media ASF:  {np.mean(asf):.4f}")

    # Etiqueta: Group (Nondemented=0, Demented=1, Converted=2)
    le = LabelEncoder()
    etiquetas = le.fit_transform(df['Group'])
    print(f"\nClases: {le.classes_}  ->  {list(range(len(le.classes_)))}")
    print(f"Distribución: {dict(zip(le.classes_, np.bincount(etiquetas)))}")

    # Armar matriz de features (8 columnas)
    X = np.column_stack([age, educ, ses, mmse, etiv, nwbv, asf, sexo])
    y = etiquetas.astype(np.int32)

    return X, y, len(le.classes_), le


def construir_modelo(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8,)),           # 8 features
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # etiquetas enteras (0,1,2)
        metrics=["accuracy"],
    )

    model.summary()
    return model


def entrenar_modelo(model, X_train, y_train, X_val, y_val):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=1,
    )
    print("\nModelo entrenado exitosamente")
    return history


def guardar_modelo(model, ruta="modelo_alzheimer.keras"):
    model.save(ruta)
    print(f"Modelo guardado en: {ruta}")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Cargar df
    X, y, num_classes, label_encoder = cargar_dataset()

    # 2. Normalizar features (media 0, std 1)
    media = X.mean(axis=0)
    std   = X.std(axis=0) + 1e-8          # evitar división por cero
    X_norm = (X - media) / std

    # 3. Split entrenamiento / validación (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} muestras | Val: {X_val.shape[0]} muestras")

    # 4. Construir modelo
    model = construir_modelo(num_classes)

    # 5. Entrenar
    history = entrenar_modelo(model, X_train, y_train, X_val, y_val)

    # 6. Evaluar
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nAccuracy en validación: {acc:.4f}  |  Loss: {loss:.4f}")

    # 7. Guardar
    guardar_modelo(model)
