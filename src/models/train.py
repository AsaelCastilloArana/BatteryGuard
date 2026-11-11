import pandas as pd
import numpy as np
import argparse
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

# Función auxiliar para graficar y guardar la matriz
def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Clase 0 (Sin Falla)', 'Clase 1 (Falla)'],
                yticklabels=['Clase 0 (Sin Falla)', 'Clase 1 (Falla)'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    
    # Guardar imagen
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'matriz_confusion_{model_name}.png')
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Gráfica guardada en: {filepath}")

def train_models(data_dir, model_dir, report_dir):
    print("[INFO] Cargando datos procesados...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    # --- MODELO 1: RANDOM FOREST ---
    print("\n[INFO] Entrenando Modelo Baseline (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    rf_f1 = f1_score(y_test, rf_preds)
    
    print(f"[RESULTADO RF] ROC-AUC: {rf_auc:.4f} | F1-Score: {rf_f1:.4f}")
    print("\n[REPORTE RF]")
    print(classification_report(y_test, rf_preds))
    
    # Generar Matriz de Confusión para RF
    plot_confusion_matrix(y_test, rf_preds, "Random_Forest", report_dir)
    
    # --- MODELO 2: XGBOOST ---
    print("\n[INFO] Entrenando Modelo Alternativo (XGBoost)...")
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, 
                        subsample=0.8, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    xgb_f1 = f1_score(y_test, xgb_preds)
    
    print(f"[RESULTADO XGB] ROC-AUC: {xgb_auc:.4f} | F1-Score: {xgb_f1:.4f}")
    print("\n[REPORTE XGB]")
    print(classification_report(y_test, xgb_preds))
    
    # Generar Matriz de Confusión para XGBoost
    plot_confusion_matrix(y_test, xgb_preds, "XGBoost", report_dir)
    
    # --- SELECCIÓN DEL GANADOR ---
    os.makedirs(model_dir, exist_ok=True)
    if xgb_auc >= rf_auc:
        print("\n[SUCCESS] XGBoost es el modelo ganador. Guardando...")
        joblib.dump(xgb, os.path.join(model_dir, 'battery_xgboost_v1.joblib'))
    else:
        print("\n[SUCCESS] Random Forest es el modelo ganador. Guardando...")
        joblib.dump(rf, os.path.join(model_dir, 'battery_rf_v1.joblib'))
    
    print(f"[SUCCESS] Modelo serializado en: {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Carpeta con datos procesados")
    parser.add_argument("--output", required=True, help="Carpeta para guardar modelo")
    parser.add_argument("--reports", required=True, help="Carpeta para guardar gráficas")
    args = parser.parse_args()
    
    train_models(args.data, args.output, args.reports)