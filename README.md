# Netflix Titles Analysis Dashboard
A Streamlit app to analyze Netflix titles (Movies vs. TV Shows) with visualizations and predictions.

## Files
- `app.py`: Streamlit app.
- `train_model.py`: Trains the model.
- `processed_netflix_titles.csv`: Dataset.
- `logistic_model.pkl`, `label_encoder.pkl`, `scaler.pkl`: Model files.
- `requirements.txt`: Dependencies.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`
3. Train model (if needed): `python train_model.py`

## Notes
- Do not share ngrok authtoken.
- Dataset: `processed_netflix_titles.csv` (or regenerate with `train_model.py`).
