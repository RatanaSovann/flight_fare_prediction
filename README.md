# Flight Fare Prediction — US Domestic Airfare

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-337AB7)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-9ACD32)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00)

Machine-learning models that predict the total fare of a US domestic flight from details a traveller knows at booking time: origin, destination, departure date and time, number of stops and the cabin class of each leg.

This repository holds the **research side** of the project: data preparation, feature engineering and model experiments. The trained models are served by a FastAPI + Streamlit application in a separate repository:

**[flight_fare_app](https://github.com/RatanaSovann/flight_fare_app)** · **[Live demo](https://at3-app-frontend-latest.onrender.com/)** · **[Project report](reports/REPORT.md)**

> The demo runs on Render's free tier. The first request after a period of inactivity can take a minute or two while the backend wakes up.

---

## Results

The dataset was split into four segments by origin airport. A model was trained for each segment, and the best model for each segment was deployed.

| Origin airports | Deployed model | Test RMSE | Test MAE |
|---|---|---|---|
| ATL, BOS, CLT, DEN | XGBoost | **$86.22** | $57.69 |
| DFW, DTW, EWR, IAD | CatBoost (Optuna-tuned) | $87.72 | $55.83 |
| JFK, LAX, LGA, MIA | XGBoost | $90.75 | — |
| OAK, ORD, PHL, SFO | LightGBM | $100.45 | — |

- **Baseline:** a mean-predicting dummy regressor scored an RMSE of about $191–204. The final models cut that error by more than half.
- **In business terms:** the median fare is about $345, so a typical error of about $85–90 is roughly 25% of a median ticket.
- **Most predictive features:** travel distance, number of stops, days between search and departure (`date_diff`) and cabin class.

The segments cover different airports, so the RMSE values are not a strict head-to-head comparison between algorithms.

## Data

- **Source:** US domestic flight search results from Expedia, covering 16 origin and 16 destination airports
- **Size:** about 40 million rows with 23 raw features. More than 60% of rows were duplicate searches and were removed.
- **Target:** `totalFare`, the total ticket price in USD, including taxes

The raw data is not included in this repository because of its size.

## Approach

The project followed CRISP-DM.

1. **Cleaning:** deduplicated repeated searches. Filled the roughly 5% of missing `totalTravelDistance` values with the median distance for each origin–destination pair.
2. **Feature engineering**
   - Calendar features from the departure date: `month`, `day`, `day_of_week`, `week_of_year`, `hour` and `minute`
   - `date_diff`: days between the search and the flight
   - `n_stops` and the cabin class of each leg (`Cabin_Leg1`–`Cabin_Leg4`), parsed from the segment strings
   - `medianTravelDistance`: a lookup table of median distance for each route, so the deployed app can supply distance from origin and destination alone
3. **Preprocessing:** scikit-learn `Pipeline` / `ColumnTransformer`. Ordinal encoding for cabin class (coach < premium coach < business < first), one-hot or target encoding for airports, and standard scaling for numeric features.
4. **Modelling:** a progression from linear models (Linear Regression, SGD, ElasticNet) to KNN and tree ensembles (Random Forest, Extra Trees, AdaBoost, XGBoost, LightGBM, CatBoost). Hyperparameters were tuned with randomised search or Optuna.
5. **Validation:** chronological or random train/validation/test splits. The models were judged on RMSE and on the gap between training and validation scores.

## Repository structure

```
├── flight_fare_prediction/   # Python package (Cookiecutter Data Science layout)
├── src/                      # Custom transformers and evaluation helpers
├── notebooks/
│   ├── dataprep*/, data_prep*/   # Data cleaning and feature engineering
│   └── *.ipynb                   # Model experiments (one notebook per algorithm)
├── models/                   # Serialised trained pipelines (.joblib / .cbm / .pkl)
├── reports/
│   ├── REPORT.md             # Condensed project report (start here)
│   └── Report.pdf            # Original full group submission
├── docs/                     # MkDocs site scaffold
├── pyproject.toml            # Poetry configuration
└── requirements.txt
```

My own work is in the notebooks prefixed `sovann_ratana-*`: data preparation, a baseline, Linear Regression, KNN, Extra Trees, LightGBM and the deployed XGBoost model.

## Getting started

```bash
git clone https://github.com/RatanaSovann/flight_fare_prediction.git
cd flight_fare_prediction
pip install -r requirements.txt   # or: poetry install
jupyter lab notebooks/
```

## Limitations and next steps

- The models don't account for seasonality, such as peak and off-peak dates or public holidays.
- They don't use layover airports or flight duration, because a user can't know these when searching.
- Each segment has its own model. A single model trained on all airports would be simpler to maintain and could share signal between routes.

## Team

This was a group project for *36120 Advanced Machine Learning Application* (Master of Data Science and Innovation, University of Technology Sydney, November 2024).

| Member | Contribution |
|---|---|
| **Ratana Sovann** | Origin segment ATL/BOS/CLT/DEN: XGBoost model. Built the deployed FastAPI backend and Streamlit frontend, integrating all four models. |
| Farhan Faiyaz | Origin segment OAK/ORD/PHL/SFO: LightGBM model |
| Yee Ting Li | Origin segment JFK/LAX/LGA/MIA: XGBoost model |
| Md Sadman Sakib | Origin segment DFW/DTW/EWR/IAD: CatBoost model |
