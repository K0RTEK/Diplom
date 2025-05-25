import pandas as pd
import numpy as np
import hdbscan

from haversine import haversine


def deg_to_rad(X):
    return np.radians(X)


def add_hdbscan_clusters(df):
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("В датафрейме должны быть колонки 'lat' и 'lon'")

    zero_coords_mask = (df['lat'] == 0) & (df['lon'] == 0)
    valid_coords_df = df[~zero_coords_mask]

    coords = valid_coords_df[['lat', 'lon']].values
    coords_rad = deg_to_rad(coords)

    # Настройка HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,  # минимальный размер кластера
        min_samples=5,  # количество соседей для точки
        metric='haversine',  # метрика Хаверсина для геокоординат
        core_dist_n_jobs=-1  # использовать все ядра
    )

    valid_coords_df.loc[:, 'geo_cluster'] = clusterer.fit_predict(coords_rad)

    df['geo_cluster'] = -2
    df.update(valid_coords_df[['geo_cluster']])

    return df


def add_geo_features(df):
    def calculate_distance(row, prev_row):
        if prev_row is None or row['carduid'] != prev_row['carduid']:
            return 0
        current_loc = (row['lat'], row['lon'])
        prev_loc = (prev_row['lat'], prev_row['lon'])
        return haversine(current_loc, prev_loc)

    df = df.sort_values(by='receiveddatetime').reset_index(drop=True)

    df['distance_prev'] = 0.0
    prev_row = None
    for i, row in df.iterrows():
        if prev_row is not None and row['carduid'] == prev_row['carduid']:
            df.at[i, 'distance_prev'] = calculate_distance(row, prev_row)
        prev_row = row

    df['time_diff_prev_hours'] = df['time_diff_prev'] / 3600
    df['speed_kmh'] = df['distance_prev'] / df['time_diff_prev_hours'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['speed_kmh'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['speed_kmh'].fillna(0, inplace=True)  # или другой метод заполнения

    df = add_hdbscan_clusters(df)

    return df


def create_rolling_features(df, windows=None):
    if windows is None:
        windows = [1, 5, 10, 20, 30, 40, 50, 60, 120, 180, 240, 480, 720, 1440]

    def rolling_counts_for_group(group):
        group = group.sort_values(by='receiveddatetime')
        timestamps = group['receiveddatetime'].astype(np.int64) // 1_000_000_000

        result_data = {f'count_{w}m': np.zeros(len(group), dtype=np.int32) for w in windows}
        time_diff_prev = np.zeros(len(group), dtype=np.float32)
        start_idx_by_window = [0] * len(windows)

        prev_time = None
        for i in range(len(group)):
            current_time = timestamps.iloc[i]

            if i == 0:
                time_diff_prev[i] = 0
            else:
                time_diff_prev[i] = current_time - prev_time

            prev_time = current_time

            for w_i, w in enumerate(windows):
                window_seconds = w * 60

                while (current_time - timestamps.iloc[start_idx_by_window[w_i]]) > window_seconds:
                    start_idx_by_window[w_i] += 1

                count_in_window = i - start_idx_by_window[w_i] + 1

                for j in range(start_idx_by_window[w_i], i + 1):
                    if result_data[f'count_{w}m'][j] < count_in_window:
                        result_data[f'count_{w}m'][j] = count_in_window

        group_result = pd.DataFrame(index=group.index)
        group_result['temp_id'] = group['temp_id']

        for k, arr in result_data.items():
            group_result[k] = arr
        group_result['time_diff_prev'] = time_diff_prev

        return group_result

    df['temp_id'] = df.index
    results = []

    for (card, term), g in df.groupby(['carduid', 'terminalid'], group_keys=False):
        feats = rolling_counts_for_group(g)
        results.append(feats)

    rolling_features = pd.concat(results, axis=0)
    rolling_features.set_index('temp_id', inplace=True)
    df = df.set_index('temp_id').join(rolling_features, rsuffix='_roll').reset_index(drop=True)

    return df
