import numpy as np

def model_metrics(df_test, model, history, x_test_scaled):
    train_final_mse = history.history['loss'][-1]
    test_mean_mse   = df_test['anomaly_score'].mean()
    recon_error_ratio = test_mean_mse / train_final_mse
    print(f"Reconstruction Error Ratio (test/train): {recon_error_ratio:.2f}")

    errors = df_test['anomaly_score']
    thr = np.quantile(errors, 0.995)
    delta = thr * 0.01  # 1% от порога
    cdf = lambda x: (errors <= x).mean()
    slope = (cdf(thr + delta) - cdf(thr - delta)) / (2 * delta)
    print(f"Slope of CDF near threshold: {slope:.3f}")

    def jaccard(a, b): return len(a & b) / len(a | b)

    idx0 = set(df_test[errors > thr].index)
    thr_lo = np.quantile(errors, 0.992)
    thr_hi = np.quantile(errors, 0.998)
    idx_lo = set(df_test[errors > thr_lo].index)
    idx_hi = set(df_test[errors > thr_hi].index)

    print("Jaccard 0.995 vs 0.992:", jaccard(idx0, idx_lo))
    print("Jaccard 0.995 vs 0.998:", jaccard(idx0, idx_hi))

    from sklearn.metrics import silhouette_score
    encoder = model.encoder

    # теперь можно получить латентные векторы
    latent = encoder.predict(x_test_scaled)
    labels = (df_test['anomaly_score'] > thr).astype(int)
    sil_score = silhouette_score(latent, labels)
    print(f"Silhouette score on latent space: {sil_score:.3f}")