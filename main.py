from src.model import save_model
from src.builder import detect_anomalies_main
from src.data_upload import load_and_preprocess_data
from src.features import (
    create_rolling_features,
    add_geo_features
)
from src.visualization import (
    plot_tsne_latent,
    plot_training_curves,
    plot_error_distribution,
    plot_reconstruction_error_distribution
)

if __name__ == '__main__':
    df = load_and_preprocess_data(r'D:\University\Diplom\FINAL\transactions.csv')
    df = create_rolling_features(df)
    df = add_geo_features(df)

    base_features = [
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
        'ticketscount'
    ]

    geo_features = ['distance_prev', 'speed_kmh', 'has_coords', 'geo_cluster']

    feature_cols = base_features + geo_features

    df_test, model, scaler, history = detect_anomalies_main(df, feature_cols)

    plot_training_curves(history)
    plot_error_distribution(df_test)
    plot_reconstruction_error_distribution(df_test)
    plot_tsne_latent(model, df_test, scaler, feature_cols)

    save_model(model, r"C:\Users\k.sidorov\PycharmProjects\Diplom\compiled_models")
