import joblib
import pandas as pd
import os

# 1. Cargar artefactos (Usamos rutas relativas para que funcione en cualquier PC)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'battery_xgboost_v1.joblib')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')

# Nota: Si el modelo ganador fue RF, cambia el MODEL_PATH a 'battery_rf_v1.joblib'
MODELO = joblib.load(MODEL_PATH)
PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

def predecir(entrada: dict) -> dict:
    # 2. Convertir a DataFrame
    df_input = pd.DataFrame([entrada])
    
    # 3. Preprocesar (OneHotEncoding)
    X_processed = PREPROCESSOR.transform(df_input[['SOH', 'Topografia', 'Clima']])
    
    # 4. Inferencia
    prediccion = int(MODELO.predict(X_processed)[0])
    probabilidad = float(MODELO.predict_proba(X_processed)[0][prediccion])
    
    # 5. Mapeo al Semáforo del Proyecto
    dictamenes = {
        0: {"etiqueta": "APROBADO", "color": "Verde"},
        1: {"etiqueta": "DENEGADO", "color": "Rojo"} # Asumiendo 1 = Falla
    }
    
    return {
        "clase_predicha": prediccion,
        "etiqueta": dictamenes[prediccion]["etiqueta"],
        "confianza": round(probabilidad * 100, 2),
        "status": "success"
    }

# Prueba de funcionamiento (se ejecuta solo si corres este archivo)
if __name__ == "__main__":
    prueba = {
        "SOH": "Degradada",
        "Topografia": "Montanosa",
        "Clima": "Calor Extremo"
    }
    print("\n[TEST] Probando función de inferencia...")
    print("Entrada:", prueba)
    print("Salida:", predecir(prueba))