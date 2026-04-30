# Machine Learning Models

Questo repository contiene script di esempio per quanto riguarda modelli di Machine Learning per fare previsioni su serie temporali (forecasting) tramite un approccio a finestra (`lookback`).

**Caratteristiche principali**
- Creazione di dataset a finestra mobile (`lookback`)
- Addestramento di un `XGBRegressor`
- Previsione iterativa (rolling) di `nfore` passi
- Visualizzazione grafica del risultato con `matplotlib`

**File principali**
- `XGBoost.py` — script Python principale.
- `BoxJenkins.csv` — dataset di esempio incluso (lo script usa per default la seconda colonna).
- `M3C_monthly.CSV` — altro dataset incluso (opzionale).
- `RandomForest.py` — script per addestrare un modello di Random Forest, usando il dataset `M3C_monthly.CSV`.
- `transformerAirlines.py` — script per addestrare un modello Transformer su un dataset di passeggeri aerei.
- `XGBoostOptuna.py` — script per ottimizzare gli iperparametri di XGBoost usando Optuna.


Requisiti
- Python 3.8+ (consigliato 3.9/3.10)
- Librerie Python: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, `torch`, `seaborn`, `optuna`.

Installazione rapida
1. Crea e attiva un virtualenv (consigliato):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```



Uso
1. Posizionati nella cartella del progetto (dove si trova `XGBoost.py`, `RandomForest.py` o `transformerAirlines.py`).

2. Esegui lo script con il dataset di default:

```bash
python "nameofthefile".py
```
