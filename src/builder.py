import pandas as pd
import numpy as np

from src.model import build_autoencoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DEFAULT_FEATURES = [
            'count_1m',
            'count_5m',
            'count_10m',
            'count_20m',
            'count_30m',
            'count_40m',
            'count_50m',
            'count_60m',
            'count_120m',
            'count_180m',
            'count_240m',
            'count_480m',
            'count_720m',
            'count_1440m',
            'time_diff_prev',
            'finalticketprice',
            'baseticketprice',
            'ticketscount',
            'distance_prev',
            'speed_kmh',
            'has_coords',
            'geo_cluster'
        ]

def prepare_data(df, cutoff_date="2025-03-21"):
    df = df.copy()
    df['date'] = df['receiveddatetime'].dt.date
    cutoff = pd.to_datetime(cutoff_date)

    df_train = df[df['receiveddatetime'] < cutoff]
    df_test = df[df['receiveddatetime'] >= cutoff]

    return df_train, df_test


def preprocess_features(df, feature_cols, max_value=1e6):
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0).values
    x = np.clip(x, a_min=None, a_max=max_value)
    return x


def scale_features(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler

def train_model(model, x_train_scaled, x_test_scaled):
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        x_train_scaled, x_train_scaled,
        epochs=50,
        batch_size=512,
        validation_data=(x_test_scaled, x_test_scaled),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return model, history


def detect_anomalies(df_test, x_test_scaled, model, threshold_quantile=0.995):
    x_test_recon = model.predict(x_test_scaled)
    mse = np.mean((x_test_scaled - x_test_recon) ** 2, axis=1)
    threshold = np.quantile(mse, threshold_quantile)

    df_test = df_test.copy().reset_index(drop=True)
    df_test['anomaly_score'] = mse
    df_test['is_anomaly'] = mse > threshold
    df_test['threshold'] = threshold

    return df_test

def save_anomalies(df_test, path):
    anomalies = df_test[df_test['is_anomaly']]
    anomalies.to_csv(path, index=False)


def detect_anomalies_main(df, result_path, feature_cols=None):
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    df_train, df_test = prepare_data(df)

    x_train = preprocess_features(df_train, feature_cols)
    x_test = preprocess_features(df_test, feature_cols)

    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

    model = build_autoencoder(x_train_scaled.shape[1])
    model, history = train_model(model, x_train_scaled, x_test_scaled)

    df_test = detect_anomalies(df_test, x_test_scaled, model)

    save_anomalies(df_test, result_path)

    return df_test, model, scaler, history, x_test_scaled
