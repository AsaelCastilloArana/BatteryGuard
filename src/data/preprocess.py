import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def preprocess_data(input_path, output_path, model_dir):
    print(f"[INFO] Cargando dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Seleccionamos las columnas de interés (SOH, Topografía, Clima y Target)
    # Nota: Mapeamos las variables numéricas a las categorías de tu proyecto
    df_clean = pd.DataFrame()
    
    # SOH (Salud de la batería)
    df_clean['SOH'] = np.where(df['state_of_health'] >= 80, 'Optima', 'Degradada')
    
    # Topografía
    df_clean['Topografia'] = df['terrain_type'].map({
        'Flat': 'Plana', 'Coastal': 'Plana', 
        'Mountainous': 'Montanosa', 'Hilly': 'Montanosa', 'Desert': 'Montanosa'
    }).fillna('Plana')
    
    # Clima (Basado en temperatura máxima)
    df_clean['Clima'] = np.where(df['maximum_temperature'] >= 40, 'Calor Extremo', 'Templado')
    
    # Variable Objetivo (Target)
    df_clean['battery_failure'] = df['battery_failure']
    
    # 2. Limpieza de nulos (por si acaso)
    df_clean = df_clean.dropna()
    
    print(f"[INFO] Registros después de limpieza: {len(df_clean)}")
    
    # 3. Separar X e y
    X = df_clean[['SOH', 'Topografia', 'Clima']]
    y = df_clean['battery_failure']
    
    # 4. Train / Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Pipeline de Preprocesamiento (OneHot para categóricas, Scaler para numéricas)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['SOH', 'Topografia', 'Clima'])
        ])
    
    # Guardar los transformadores
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
    
    # Procesar y guardar
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Guardar sets procesados
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(X_train_processed.toarray()).to_csv(os.path.join(os.path.dirname(output_path), 'X_train.csv'), index=False)
    pd.DataFrame(X_test_processed.toarray()).to_csv(os.path.join(os.path.dirname(output_path), 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(os.path.dirname(output_path), 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(os.path.dirname(output_path), 'y_test.csv'), index=False)
    
    print("[SUCCESS] Preprocesamiento completado. Datos guardados en data/processed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ruta al CSV original")
    parser.add_argument("--output", required=True, help="Ruta de salida de datos procesados")
    parser.add_argument("--model_dir", required=True, help="Carpeta para guardar preprocesadores")
    args = parser.parse_args()
    
    preprocess_data(args.data, args.output, args.model_dir)