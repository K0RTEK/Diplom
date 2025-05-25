import folium
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_clusters_on_map(df):
    m = folium.Map(location=[55.0, 82.9], zoom_start=12)

    for cluster_id in df['geo_cluster'].unique():
        if cluster_id == -2:
            continue
        cluster_data = df[df['geo_cluster'] == cluster_id]
        color = 'red' if cluster_id == -1 else 'blue'  # -1 = шум в HDBSCAN

        folium.Marker(
            location=[cluster_data['lat'].mean(), cluster_data['lon'].mean()],
            popup=f"Кластер {cluster_id}, {len(cluster_data)} записей",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)

        for _, row in cluster_data.head(1000).iterrows():  # ограничение для производительности
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3 if cluster_id != -1 else 5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)

    return m


def plot_training_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.title('Кривые обучения автоэнкодера')
    plt.ylim(0, max(history.history['loss'] + history.history['val_loss']) * 1.1)
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show()


def plot_error_distribution(df_test):
    errors = df_test['anomaly_score']
    p99_9 = np.quantile(errors, 0.999)
    errors_trunc = errors.clip(upper=p99_9)

    plt.figure(figsize=(8, 5))
    plt.hist(errors_trunc, bins=100, density=True, alpha=0.6, label='All (<=99.9%)')
    sorted_e = np.sort(errors_trunc)
    cdf = np.arange(len(sorted_e)) / len(sorted_e)
    ax2 = plt.gca().twinx()
    ax2.plot(sorted_e, cdf, color='black', linestyle='--', label='CDF')
    plt.xscale('log')
    plt.xlabel('Ошибка реконструкции (MSE)')
    plt.ylabel('Плотность / CDF')
    plt.title('Распределение ошибок и CDF (лог‐шкала)')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show()


def plot_tsne_latent(model, df_test, scaler, feature_cols):
    x_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    x_test_scaled = scaler.transform(x_test)

    encoder = model.encoder
    latent = encoder.predict(x_test_scaled)
    tsne = TSNE(n_components=2, random_state=42)
    z = tsne.fit_transform(latent)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=df_test['is_anomaly'], palette=['green', 'red'], alpha=0.6)
    plt.title('t-SNE латентного пространства (0 – норма, 1 – аномалия)')
    plt.show()


def plot_reconstruction_error_distribution(df_test):
    thr = np.quantile(df_test['anomaly_score'], 0.995)
    norm_errors = df_test['anomaly_score'][df_test['anomaly_score'] <= thr]
    anom_errors = df_test['anomaly_score'][df_test['anomaly_score'] > thr]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    ax.hist(norm_errors, bins=100, alpha=0.7, label=f'Нормальные ({len(norm_errors)})')
    ax.hist(anom_errors, bins=20, alpha=0.9, label=f'Аномальные ({len(anom_errors)})')
    ax.axvline(thr, color='k', linestyle='--', linewidth=2, label=f'Порог ({thr:.2f})')
    ax.set_xscale('log')
    ax.set_xlim(norm_errors.min() * 0.8, max(anom_errors.max() * 1.2, thr * 10))
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frequency')
    ax.set_title('Распределение ошибок реконструкции')
    ax.legend(loc='upper left')
    ax.vlines(anom_errors, ymin=0, ymax=ax.get_ylim()[1] * 0.02, alpha=0.3, linewidth=0.5)

    ax_ins = inset_axes(ax,
                        width="30%", height="30%",
                        loc='upper right', borderpad=1.4)

    x_max = anom_errors.max() * 1.1
    bins = np.logspace(np.log10(thr), np.log10(x_max), 500)

    ax_ins.hist(anom_errors, bins=bins, alpha=0.9)
    ax_ins.axvline(thr, color='k', linestyle='--', linewidth=1)

    ax_ins.set_xscale('log')
    ax_ins.set_xlim(thr, x_max)

    counts, _ = np.histogram(anom_errors, bins=bins)
    ax_ins.set_ylim(0, counts.max() * 1.2)

    ax_ins.set_title('Приближение: аномалии', fontsize=8)
    ax_ins.tick_params(axis='both', which='major', labelsize=6)

    plt.show()
