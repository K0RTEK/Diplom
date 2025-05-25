import pandas as pd
import numpy as np

from src.model import build_autoencoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def prepare_data(df, cutoff_date="2025-03-21"):
    df = df.copy()
    df['date'] = df['receiveddatetime'].dt.date
    cutoff = pd.to_datetime(cutoff_date)

    df_train = df[df['receiveddatetime'] < cutoff]
    df_test = df[df['receiveddatetime'] >= cutoff]

    return df_train, df_test


def preprocess_features(df, feature_cols, max_value=1e6):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0).values
    X = np.clip(X, a_min=None, a_max=max_value)
    return X


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, output_layer)
    return autoencoder


def train_model(model, X_train_scaled, X_test_scaled):
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
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=512,
        validation_data=(X_test_scaled, X_test_scaled),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return model, history


def detect_anomalies(df_test, X_test_scaled, model, threshold_quantile=0.995):
    X_test_recon = model.predict(X_test_scaled)
    mse = np.mean((X_test_scaled - X_test_recon) ** 2, axis=1)
    threshold = np.quantile(mse, threshold_quantile)

    df_test = df_test.copy().reset_index(drop=True)
    df_test['anomaly_score'] = mse
    df_test['is_anomaly'] = mse > threshold
    df_test['threshold'] = threshold

    return df_test


def detect_anomalies_main(df, feature_cols):
    df_train, df_test = prepare_data(df)

    X_train = preprocess_features(df_train, feature_cols)
    X_test = preprocess_features(df_test, feature_cols)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = build_autoencoder(X_train_scaled.shape[1])
    model, history = train_model(model, X_train_scaled, X_test_scaled)

    df_test = detect_anomalies(df_test, X_test_scaled, model)

    return df_test, model, scaler, history
