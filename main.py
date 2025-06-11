import logging
import time

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()

    logger.info("Шаг 1: Загрузка и предобработка данных")
    df = load_and_preprocess_data(r'/Users/kirill/PycharmProjects/Diplom/data/raw/transactions.csv')

    logger.info("Шаг 2: Создание скользящих признаков")
    df = create_rolling_features(df)

    logger.info("Шаг 3: Добавление гео-признаков")
    df = add_geo_features(df)

    logger.info("Шаг 4: Детектирование аномалий и разделение тестовой выборки")
    df_test, model, scaler, history, x_test_scaled = detect_anomalies_main(
        df, r"/Users/kirill/PycharmProjects/Diplom/data/processed/anomalies.csv"
    )

    logger.info("Шаг 5: Визуализация кривых обучения")
    plot_training_curves(history)
    logger.info("plot_training_curves завершён")

    logger.info("Шаг 6: Визуализация распределения ошибок")
    plot_error_distribution(df_test)
    logger.info("plot_error_distribution завершён")

    logger.info("Шаг 7: Визуализация распределения ошибок реконструкции")
    plot_reconstruction_error_distribution(df_test)
    logger.info("plot_reconstruction_error_distribution завершён")

    logger.info("Шаг 8: TSNE визуализация латентного пространства")
    plot_tsne_latent(model, df_test, scaler, DEFAULT_FEATURES)
    logger.info("plot_tsne_latent завершён")

    logger.info("Шаг 9: Оценка метрик модели")
    model_metrics(df_test, model, history, x_test_scaled)
    logger.info("model_metrics завершён")

    logger.info("Шаг 10: Сохранение модели")
    save_model(model, r"D:\Work\Diplom\compiled_models\model.keras")
    logger.info("Модель сохранена")

    elapsed = time.time() - start_time
    logger.info(f"Общее время выполнения программы: {elapsed:.2f} секунд")

if __name__ == "__main__":
    main()