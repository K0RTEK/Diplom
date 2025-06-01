import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Анализ аномалий",
    layout="wide"
)

st.title("Анализ аномалий, найденных моделью")

# 1. Загрузка данных
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV с результатами модели (должны быть столбцы: lat, lon, date, anomaly_score, is_anomaly и пр.)",
    type=["csv"]
)

if uploaded_file is not None:
    # Попытка прочитать файл как CSV
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["receiveddatetime", "transactiontime", "tarifficationdatetime", "date"])
    except Exception as e:
        st.error(f"Не удалось загрузить данные: {e}")
        st.stop()

    # Проверяем наличие обязательных столбцов
    required_cols = {"lat", "lon", "date", "anomaly_score", "is_anomaly"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"В файле отсутствуют обязательные столбцы: {missing}")
        st.stop()

    # 2. Препроцессинг (если нужно)
    # Убедимся, что столбец is_anomaly булев
    if df["is_anomaly"].dtype != bool:
        df["is_anomaly"] = df["is_anomaly"].astype(bool)

    # Добавим вспомогательные столбцы: год, месяц, день
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # 3. Sidebar фильтры
    st.sidebar.header("Фильтры")
    # Фильтр по дате
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "Выберите диапазон дат",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    # Преобразуем date_range к datetime
    start_date = datetime.combine(date_range[0], datetime.min.time())
    end_date = datetime.combine(date_range[1], datetime.max.time())

    # Фильтр по порогу anomaly_score (пицельзависимый)
    min_score = float(df["anomaly_score"].min())
    max_score = float(df["anomaly_score"].max())
    score_threshold = st.sidebar.slider(
        "Порог anomaly_score",
        min_value=min_score,
        max_value=max_score,
        value=float(df["threshold"].median()) if "threshold" in df.columns else min_score,
        step=(max_score - min_score) / 100.0
    )

    # Фильтр по geo_cluster, если есть
    geo_unique = sorted(df["geo_cluster"].unique())
    selected_clusters = st.sidebar.multiselect(
        "Выберите geo_cluster (оставьте пустым для всех)",
        options=geo_unique,
        default=geo_unique
    )

    # 4. Применяем фильтры
    mask = (
        (df["date"] >= start_date) &
        (df["date"] <= end_date) &
        (df["anomaly_score"] >= score_threshold) &
        (df["geo_cluster"].isin(selected_clusters))
    )
    filtered_df = df.loc[mask].copy()

    # 5. Основные метрики
    st.subheader("Основные метрики")
    total_count = len(filtered_df)
    anomaly_count = int(filtered_df["is_anomaly"].sum())
    normal_count = total_count - anomaly_count
    anomaly_pct = (anomaly_count / total_count * 100) if total_count > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Записей (после фильтрации)", total_count)
    col2.metric("Аномалий", anomaly_count)
    col3.metric("Нормальных", normal_count)
    col4.metric("Доля аномалий, %", f"{anomaly_pct:.2f}")

    # 6. Гистограмма распределения anomaly_score
    st.subheader("Распределение anomaly_score")
    fig_score = px.histogram(
        filtered_df,
        x="anomaly_score",
        color="is_anomaly",
        nbins=50,
        title="Гистограмма anomaly_score (цветом: аномалия/норма)",
        labels={"anomaly_score": "anomaly_score", "count": "Количество"},
        marginal="box"
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # 7. Временная динамика аномалий
    st.subheader("Динамика числа аномалий по дням")
    # Группируем по дате
    df_time = filtered_df.groupby(filtered_df["date"].dt.date)["is_anomaly"].agg(["sum", "count"])
    df_time = df_time.rename(columns={"sum": "anomaly_count", "count": "total_count"})
    df_time["normal_count"] = df_time["total_count"] - df_time["anomaly_count"]
    df_time = df_time.reset_index().rename(columns={"date": "date_only"})
    # Линейный график: аномалии и нормальные
    fig_time = px.line(
        df_time,
        x="date_only",
        y=["anomaly_count", "normal_count"],
        title="Число аномалий и нормальных записей по дате",
        labels={"value": "Количество", "date_only": "Дата", "variable": "Тип"}
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # 8. Карта: географическое распределение аномалий
    st.subheader("Географическое распределение (карта)")
    # Для работы st.map нужны столбцы lat и lon
    map_df = filtered_df[["lat", "lon", "is_anomaly"]].copy()
    # Для наглядности: отдельно точки аномалий и нормальных
    anomalies_map = map_df[map_df["is_anomaly"]]
    normals_map = map_df[~map_df["is_anomaly"]]

    st.markdown("**Аномалии**")
    if not anomalies_map.empty:
        st.map(anomalies_map.rename(columns={"lat": "latitude", "lon": "longitude"}))
    else:
        st.write("Аномалий не найдено в выбранных фильтрах.")

    st.markdown("**Нормальные записи**")
    if not normals_map.empty:
        st.map(normals_map.rename(columns={"lat": "latitude", "lon": "longitude"}))
    else:
        st.write("Нормальных записей не найдено в выбранных фильтрах.")

    # 9. Таблица аномалий
    st.subheader("Таблица аномалий")
    anomalies_table = filtered_df[filtered_df["is_anomaly"]].copy()
    if not anomalies_table.empty:
        # Покажем ключевые столбцы для быстрого обзора
        cols_to_show = [
            "terminalid", "date", "lat", "lon", "finalticketprice",
            "speed_kmh", "geo_cluster", "anomaly_score", "threshold"
        ]
        cols_to_show = [c for c in cols_to_show if c in anomalies_table.columns]
        st.dataframe(anomalies_table[cols_to_show].sort_values(by="anomaly_score", ascending=False))
    else:
        st.write("Нет аномальных записей для отображения.")

    # 10. Дополнительный анализ: сравнение признаков аномалий vs нормальных
    st.subheader("Сравнение распределения признаков у аномалий и нормальных")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Исключим идентификаторы и счётчики, оставим только несколько ключевых метрик
    compare_cols = [c for c in ["speed_kmh", "finalticketprice", "distance_prev", "time_diff_prev_hours"] if c in numeric_cols]
    if compare_cols:
        for col in compare_cols:
            fig_box = px.box(
                filtered_df,
                x="is_anomaly",
                y=col,
                points="all",
                title=f"Распределение {col} для аномалий и нормальных записей",
                labels={"is_anomaly": "Аномалия (True/False)", col: col}
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.write("Нет числовых столбцов для сравнения.")

    # 11. Возможность скачать выборку аномалий
    st.subheader("Скачать выборку аномалий")
    if not anomalies_table.empty:
        csv_anomalies = anomalies_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Скачать CSV с аномалиями",
            data=csv_anomalies,
            file_name="anomalies.csv",
            mime="text/csv"
        )
    else:
        st.write("Нет данных для скачивания.")

else:
    st.info("Пожалуйста, загрузите CSV-файл с данными, чтобы начать анализ.")
