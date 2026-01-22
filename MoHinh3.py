# =========================
# HỒI QUY ĐA BIẾN
# Sử dụng: TiSuatSinhTho, TiSuatTuTho, MatDoDS
# Dự báo: DanSoTB (năm chọn)
# =========================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =========================
# 1) Đọc dữ liệu
# =========================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Nam"] = df["Nam"].astype(int)

    # Quy đổi dân số: nghìn người → người
    df["DanSoTB"] = df["DanSoTB"] * 1000
    return df


# =========================
# 2) Hồi quy đa biến + đánh giá + dự báo năm chọn
# =========================
def multivariate_model_forecast(
    df: pd.DataFrame,
    tinh: str,
    nam_chon: int,
    test_years: int = 2
):
    """
    Mô hình hồi quy đa biến:
        DanSoTB = b0
                  + b1 * TiSuatSinhTho
                  + b2 * TiSuatTuTho
                  + b3 * MatDoDS

    - Lọc theo tỉnh
    - Chỉ dùng dữ liệu ≤ năm chọn
    - Chia train/test theo thời gian
    - Chuẩn hóa dữ liệu
    - Đánh giá MAE, RMSE, R2
    - Dự báo DUY NHẤT cho năm nam_chon
    """

    # --- Lọc theo tỉnh ---
    g = df[df["Tinh"] == tinh].sort_values("Nam").copy()
    if g.empty:
        raise ValueError(f"Không tìm thấy tỉnh: {tinh}")

    # --- Giới hạn dữ liệu đến năm chọn ---
    g = g[g["Nam"] <= nam_chon].copy()

    features = ["TiSuatSinhTho", "TiSuatTuTho", "MatDoDS"]
    target = "DanSoTB"

    g = g.dropna(subset=features + [target])

    if len(g) < test_years + 2:
        raise ValueError("Không đủ dữ liệu để chia train/test.")

    # --- Chia train / test theo thời gian ---
    train = g.iloc[:-test_years].copy()
    test = g.iloc[-test_years:].copy()

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # --- Chuẩn hóa dữ liệu ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)
    ).ravel()

    # --- Huấn luyện mô hình ---
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    # --- Dự đoán & đánh giá trên tập test ---
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    ).ravel()

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred) if len(y_test) >= 2 else np.nan

    # --- Dự báo cho năm nam_chon ---
    row_target = g[g["Nam"] == nam_chon]
    X_forecast = scaler_X.transform(row_target[features])
    y_forecast_scaled = model.predict(X_forecast)
    du_bao_nam_chon = scaler_y.inverse_transform(
        y_forecast_scaled.reshape(-1, 1)
    )[0, 0]

    # --- Kết quả ---
    summary_text = (
        f"Tỉnh: {tinh}\n"
        f"Năm dự báo: {nam_chon}\n"
        f"Mô hình hồi quy đa biến:\n"
        f"DanSoTB = b0 + b1*TiSuatSinhTho + b2*TiSuatTuTho + b3*MatDoDS\n"
        f"Đánh giá (test {test_years} năm cuối):\n"
        f"  - MAE : {mae:,.2f}\n"
        f"  - RMSE: {rmse:,.2f}\n"
        f"  - R2  : {r2:.4f}\n"
        f"Dự báo dân số năm {nam_chon}: {du_bao_nam_chon:,.0f} người\n"
    )

    return {
        "summary_text": summary_text,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "forecast": {
            "nam_du_bao": nam_chon,
            "du_bao": du_bao_nam_chon
        },
        "model": model
    }


# =========================
# 3) Chạy thử
# =========================
if __name__ == "__main__":
    df = load_data("DataDanSoVN.csv")

    kq = multivariate_model_forecast(
        df,
        tinh="HaNoi",
        nam_chon=2023,
        test_years=2
    )

    print(kq["summary_text"])
