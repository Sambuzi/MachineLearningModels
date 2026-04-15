# Machine Learning Models

Questo repository contiene script di esempio per quanto riguarda modelli di Machine Learning per fare previsioni su serie temporali (forecasting) tramite un approccio a finestra (`lookback`). Lo script principale attuale è `XGBoost.py`.

**Caratteristiche principali**
- Creazione di dataset a finestra mobile (`lookback`)
- Addestramento di un `XGBRegressor`
- Previsione iterativa (rolling) di `nfore` passi
- Visualizzazione grafica del risultato con `matplotlib`

**File principali**
- `XGBoost.py` — script Python principale.
- `BoxJenkins.csv` — dataset di esempio incluso (lo script usa per default la seconda colonna).
- `M3C_monthly.CSV` — altro dataset incluso (opzionale).

Requisiti
- Python 3.8+ (consigliato 3.9/3.10)
- Pacchetti Python elencati in `requirements.txt`.

Installazione rapida
1. Crea e attiva un virtualenv (consigliato):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Uso
1. Posizionati nella cartella del progetto (dove si trova `XGBoost.py`).

2. Esegui lo script con il dataset di default:

```bash
python XGBoost.py
```
