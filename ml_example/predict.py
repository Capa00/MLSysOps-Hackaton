import pandas as pd
import numpy as np
import joblib


def predict(model_path, df):
    RESAMPLE_FREQ = "1T"     
    FORECAST_HORIZON = 10 
    bundle = joblib.load(model_path)
    

    if "time" in df.columns:
        fx = df.copy()
        fx["time"] = pd.to_datetime(df["time"])
        fx = df.set_index("time").sort_index()
        #fx = fx.sort_values("time")

    pipe, meta = bundle["pipeline"], bundle["meta"]
    feature_cols = meta["feature_cols"]; LAGS = meta["lags"]; K = max(LAGS)
    target = meta["target"]; exog_cols = meta["exog_cols"]

    #fx = future_exog_df.copy()
    fx.index = pd.DatetimeIndex(fx.index)
    missing = [c for c in exog_cols if c not in fx.columns]
    if missing:
        raise ValueError(f"future_exog missing columns: {missing}")
    fx = fx[exog_cols].astype(float)

    hist = np.array(meta["last_k_target"], dtype=float)
    need_minute = "minute" in feature_cols
    need_hour   = "hour"   in feature_cols
    need_dow    = "dow"    in feature_cols

    preds = []
    for t, row_exog in fx.iterrows():
        row = {}
        for L in LAGS:
            row[f"{target}_lag{L}"] = hist[-L]
        for c in exog_cols:
            row[c] = float(row_exog[c])
        if need_minute: row["minute"] = t.minute
        if need_hour:   row["hour"]   = t.hour
        if need_dow:    row["dow"]    = t.dayofweek
        X_pred = pd.DataFrame([row], index=[t])[feature_cols]
        y_hat = float(pipe.predict(X_pred)[0])
        hist = np.append(hist, y_hat)[-K:]
        preds.append({"Time": t, target: y_hat})
    return pd.DataFrame(preds).set_index("Time")
