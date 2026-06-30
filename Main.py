
# Weather LSTM Forecasting for weatherHistory.csv
import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------- LOAD ----------------
df = pd.read_csv("weatherHistory.csv")
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
df = df.sort_values("Formatted Date").set_index("Formatted Date")

features = [
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Pressure (millibars)",
    "Visibility (km)"
]
df = df[features].dropna()

print(df.head())
print(df.describe())

# ---------------- SCALE ----------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)

SEQ_LEN = 168   # one week of hourly data

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len,0])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, SEQ_LEN)

train_end = int(len(X)*0.70)
val_end = int(len(X)*0.85)

X_train,y_train = X[:train_end],y[:train_end]
X_val,y_val = X[train_end:val_end],y[train_end:val_end]
X_test,y_test = X[val_end:],y[val_end:]

# ---------------- MODEL ----------------
tf.random.set_seed(42)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN,X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

early = EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True)

history = model.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early],
    verbose=1
)

def inverse_temp(vals):
    dummy = np.zeros((len(vals), len(features)))
    dummy[:,0]=vals
    return scaler.inverse_transform(dummy)[:,0]

pred = model.predict(X_test, verbose=0).flatten()
pred = inverse_temp(pred)
actual = inverse_temp(y_test)

mae = mean_absolute_error(actual,pred)
rmse = np.sqrt(mean_squared_error(actual,pred))
r2 = r2_score(actual,pred)

print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2  : {r2:.3f}")

# ---------------- FUTURE ----------------
last = scaled[-SEQ_LEN:].copy()
future_scaled=[]

for _ in range(30):
    p = model.predict(last.reshape(1,SEQ_LEN,len(features)), verbose=0)[0,0]
    future_scaled.append(p)
    row = last[-1].copy()
    row[0]=p
    last=np.vstack((last[1:],row))

future = inverse_temp(np.array(future_scaled))
future_dates = pd.date_range(df.index[-1]+pd.Timedelta(hours=1), periods=30, freq="h")

# ---------------- PLOTS ----------------
DARK='#0f1117'; PANEL='#1a1f2e'
TEXT='#e2e8f0'; DIM='#64748b'
BLUE='#3b82f6'; GREEN='#22c55e'; GOLD='#f59e0b'; RED='#ef4444'

def style(ax,title):
    ax.set_facecolor(PANEL)
    ax.set_title(title,color=TEXT)
    ax.tick_params(colors=DIM)
    ax.grid(alpha=.3)

fig=plt.figure(figsize=(18,12),facecolor=DARK)
gs=gridspec.GridSpec(2,2)

dates=df.index[val_end+SEQ_LEN:]

ax=fig.add_subplot(gs[0,:]);style(ax,"Actual vs Predicted")
ax.plot(dates,actual,label="Actual")
ax.plot(dates,pred,'--',label="Pred")
ax.legend()

ax=fig.add_subplot(gs[1,0]);style(ax,"Loss")
ax.plot(history.history["loss"])
ax.plot(history.history["val_loss"])
ax.set_yscale("log")

ax=fig.add_subplot(gs[1,1]);style(ax,"30-step Forecast")
ax.plot(df.index[-200:],df["Temperature (C)"].values[-200:],label="History")
ax.plot(future_dates,future,'--',label="Forecast")
ax.legend()

plt.tight_layout()
plt.savefig("lstm_weather.png",dpi=150)

model.save("weather_lstm.keras")
with open("weather_scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

print("Done.")
