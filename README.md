# Weather Forecasting Using LSTM

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network for time-series weather forecasting using historical weather data. The model learns temporal patterns from multiple weather attributes and predicts future temperature values. It also generates performance metrics, visualizations, and a short-term temperature forecast.

## Features

- Multi-feature weather time-series forecasting
- LSTM-based deep learning model built with TensorFlow/Keras
- Data preprocessing with MinMax scaling
- Sequential data generation using sliding windows
- Training, validation, and test dataset split
- Early stopping to prevent overfitting
- Model evaluation using MAE, RMSE, and R² Score
- Future temperature prediction for the next 30 hours
- Automatic visualization of predictions and training history
- Saves trained model and scaler for future inference

## Dataset

The project expects a dataset named:

```
weatherHistory.csv
```

Required columns:

- Formatted Date
- Temperature (C)
- Humidity
- Wind Speed (km/h)
- Pressure (millibars)
- Visibility (km)

The dataset is sorted chronologically before training.

## Project Structure

```
.
├── weather_lstm_weatherHistory.py
├── weatherHistory.csv
├── weather_lstm.keras          # Generated after training
├── weather_scaler.pkl          # Generated after training
├── lstm_weather.png            # Generated visualization
└── README.md
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pickle

## Model Architecture

- LSTM (128 units, return sequences)
- Dropout (0.2)
- LSTM (64 units)
- Dropout (0.2)
- Dense (32, ReLU)
- Dense (1 output)

Loss Function:
- Mean Squared Error (MSE)

Optimizer:
- Adam

Early stopping is used with validation loss monitoring.

## Data Pipeline

1. Load weather dataset
2. Parse and sort timestamps
3. Select relevant weather features
4. Normalize data using MinMaxScaler
5. Create hourly sequences of length 168 (one week)
6. Split data:
   - 70% Training
   - 15% Validation
   - 15% Testing
7. Train the LSTM model
8. Evaluate model performance
9. Predict the next 30 hourly temperatures
10. Save model, scaler, and plots

## Evaluation Metrics

The model reports:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Generated Outputs

After execution, the following files are created:

- `weather_lstm.keras` — Trained LSTM model
- `weather_scaler.pkl` — Saved MinMax scaler
- `lstm_weather.png` — Visualization containing:
  - Actual vs Predicted Temperature
  - Training & Validation Loss
  - 30-Hour Temperature Forecast

## Installation

Install the required packages:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

## Running the Project

Place `weatherHistory.csv` in the project directory and execute:

```bash
python weather_lstm_weatherHistory.py
```

## Future Improvements

- Predict multiple weather parameters simultaneously
- Hyperparameter optimization
- Bidirectional LSTM or GRU architectures
- Attention mechanisms
- Longer forecasting horizons
- Interactive dashboard for visualization
- Model deployment using Flask or FastAPI

## License

This project is intended for educational and research purposes.
