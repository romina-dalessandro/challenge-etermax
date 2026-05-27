# Predicción de Churn D1 — Trivia Crack

Challenge para Etermax. El objetivo es predecir qué usuarios que instalan Trivia Crack hoy no van a volver a jugar al día siguiente (churn D1).

---

## Problema

Clasificación binaria: dado un usuario que instaló el juego en el día 0 (D0), predecir si va a churnar al día siguiente (D1).

- **Target 1**: el usuario no jugó al día siguiente → churnó
- **Target 0**: el usuario volvió a jugar → retuvo

El dataset cubre ~18.500 usuarios de Argentina durante 8 días de junio-julio 2018, con distribución balanceada (54% churn / 46% retención).

---

## Estructura del proyecto

```
challenge_etermax/
├── data/
│   ├── raw/               # Datos originales y parquet extraído
│   ├── processed/         # Dataset transformado y modelo serializado
│   └── splits/            # Train/test splits listos para modelar
├── docs/
│   └── Ejercicio Data Science - 26.pdf
└── notebooks/
    ├── 01_EDA.ipynb                  # Análisis exploratorio
    ├── 02_extract.ipynb              # ETL — Extract
    ├── 03_transform.ipynb            # ETL — Transform
    ├── 04_load.ipynb                 # ETL — Load / train-test split
    ├── 05_model_baseline.ipynb       # Logistic Regression
    ├── 06_model_random_forest.ipynb  # Random Forest
    ├── 07_model_xgboost.ipynb        # XGBoost base
    └── 08_model_xgboost_tuned.ipynb  # XGBoost + CV + tuning + SHAP
```

---

## Pipeline

### ETL
1. **Extract** (`02`): lectura del CSV, validación de calidad, deduplicación, exportación a parquet.
2. **Transform** (`03`): feature engineering sobre el dataset limpio.
3. **Load** (`04`): train/test split estratificado (80/20), exportación de splits.

### Feature Engineering
| Feature | Descripción |
|---|---|
| `event_1_log` … `event_5_log` | Log1p de los 5 eventos de D0 (reduce efecto outliers) |
| `event_3_used` | Binaria: ¿realizó el evento 3? (74% zeros en el original) |
| `install_hour` | Hora de instalación en UTC |
| `install_dow` | Día de la semana (0=lunes … 6=domingo) |
| `is_weekend` | Binaria: instaló en fin de semana |
| `platform_enc` | Binaria: iOS=1, Android=0 |
| `gender_enc` | Label encoding: female=0, male=1, unknown=2 |
| `is_teen` | Binaria: rango etario mínimo = 13 años |
| `region_enc` | Label encoding de country_region |

---

## Modelos

| Modelo | AUC Test |
|---|---|
| Logistic Regression (baseline) | 0.7749 |
| Random Forest | 0.7731 |
| XGBoost base | 0.7941 |
| **XGBoost tuned** | **0.7948** |

El modelo final es un XGBoost optimizado con `RandomizedSearchCV` (50 iteraciones, 5-fold stratified CV).

**Parámetros óptimos:** `max_depth=3`, `learning_rate=0.05`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=0.6`

**AUC CV:** 0.7972 ± 0.007 — sin overfitting, el modelo generaliza bien.

---

## Hallazgos principales

- **El comportamiento en D0 es el mejor predictor de churn**: los usuarios que más interactúan en su primer día tienen mucho menor probabilidad de churnar.
- **El momento de instalación agrega señal**: instalar un sábado tiene 73% de churn vs. 54% promedio; instalar de madrugada (2-3am hora Argentina) también concentra churn.
- **iOS churna 14 puntos menos que Android** (43% vs. 57%).
- **Las features demográficas aportan poco** al modelo final — el comportamiento domina sobre el perfil.

### Features más importantes (SHAP)
1. `event_4_log` — valores altos reducen fuertemente el churn
2. `install_dow` — fin de semana aumenta el riesgo
3. `event_1_log` — más actividad en D0, menos churn
4. `event_3_used` — feature binaria construida en el ETL, cuarto lugar

---

## Reproducción

```bash
# Instalar dependencias
pip install pandas numpy scikit-learn xgboost shap matplotlib joblib pyarrow

# Correr en orden
jupyter notebook notebooks/02_extract.ipynb
jupyter notebook notebooks/03_transform.ipynb
jupyter notebook notebooks/04_load.ipynb
jupyter notebook notebooks/08_model_xgboost_tuned.ipynb
```

> El dataset original (`data/raw/Dataset DS - 26.csv`) no está incluido en el repositorio.
