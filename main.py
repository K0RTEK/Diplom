from src.model import save_model
from src.metrics import model_metrics
from src.builder import detect_anomalies_main, DEFAULT_FEATURES
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
    df = load_and_preprocess_data(r'D:\Work\Diplom\data\raw\transactions.csv')
    df = create_rolling_features(df)
    df = add_geo_features(df)
    df_test, model, scaler, history, x_test_scaled = detect_anomalies_main(df, r"D:\Work\Diplom\data\processed\anomalies.csv")
    plot_training_curves(history)
    plot_error_distribution(df_test)
    plot_reconstruction_error_distribution(df_test)
    plot_tsne_latent(model, df_test, scaler, DEFAULT_FEATURES)
    model_metrics(df_test, model, history, x_test_scaled)
    save_model(model, r"D:\Work\Diplom\compiled_models\model.keras")