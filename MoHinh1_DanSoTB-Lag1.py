
# HỒI QUY ĐƠN BIẾN ĐƠN BIẾN
# Dự báo DanSoTB bằng cách sd dân số năm trước(lag1)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1) Đọc dữ liệu
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Nam"] = df["Nam"].astype(int)

    # Quy đổi nghìn người → người
    df["DanSoTB"] = df["DanSoTB"] * 1000
    return df

# 2) Đơn biến AR(1): train + test + dự báo năm chọn
def ar1_train_test_forecast(
    df: pd.DataFrame,
    tinh: str,
    nam_chon: int,
    test_years: int = 2
):
    """
    Mô hình AR(1):
        DanSoTB_t = b0 + b1 * DanSoTB_(t-1)

    - Lọc theo tỉnh
    - Tạo biến trễ lag-1
    - Chuẩn hóa dữ liệu
    - Chia train/test theo thời gian
    - Đánh giá MAE, RMSE, R2
    - Dự báo DUY NHẤT cho năm nam_chon
    """

    # --- Lọc dữ liệu theo tỉnh ---
    g = df[df["Tinh"] == tinh].sort_values("Nam").copy()
    if g.empty:
        raise ValueError(f"Không tìm thấy tỉnh: {tinh}")

    # --- Tạo biến trễ ---
    g["DanSoTB_lag1"] = g["DanSoTB"].shift(1)
    gl = g.dropna(subset=["DanSoTB_lag1"]).reset_index(drop=True)

    # --- Giới hạn dữ liệu đến năm chọn ---
    gl_upto = gl[gl["Nam"] <= nam_chon].copy()
    if len(gl_upto) < test_years + 2:
        raise ValueError("Không đủ dữ liệu để chia train/test.")

    # --- Chia train / test theo thời gian ---
    train = gl_upto.iloc[:-test_years].copy()
    test = gl_upto.iloc[-test_years:].copy()

    X_train = train[["DanSoTB_lag1"]]
    y_train = train["DanSoTB"]

    X_test = test[["DanSoTB_lag1"]]
    y_test = test["DanSoTB"]

    # --- Chuẩn hóa ---
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

    # --- Đánh giá trên tập test ---
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    ).ravel()

    test["DanSoDuBao_test"] = y_test_pred

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred) if len(test) >= 2 else np.nan

    # --- Dự báo cho năm nam_chon ---
    row_target = gl_upto[gl_upto["Nam"] == nam_chon]
    if row_target.empty:
        raise ValueError(f"Không có dữ liệu lag cho năm {nam_chon}.")

    X_fc = scaler_X.transform(row_target[["DanSoTB_lag1"]])
    y_fc_scaled = model.predict(X_fc)
    du_bao_nam_chon = scaler_y.inverse_transform(
        y_fc_scaled.reshape(-1, 1)
    )[0, 0]

    # --- Kết quả ---
    summary_text = (
        f"Tỉnh: {tinh}\n"
        f"Năm dự báo: {nam_chon}\n"
        f"Mô hình: DanSoTB_t = b0 + b1 * DanSoTB_(t-1)\n"
        f"Đánh giá (test {test_years} năm cuối):\n"
        f"  - MAE : {mae:,.2f}\n"
        f"  - RMSE: {rmse:,.2f}\n"
        f"  - R2  : {r2:.4f}\n"
        f"Dự báo dân số năm {nam_chon}: {du_bao_nam_chon:,.0f} người\n"
    )

    return {
        "summary_text": summary_text,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "test_table": test[["Nam", "DanSoTB_lag1", "DanSoTB", "DanSoDuBao_test"]],
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

    kq = ar1_train_test_forecast(
        df,
        tinh="HaNoi",
        nam_chon=2023,
        test_years=2
    )

    print(kq["summary_text"])
    print("BẢNG TEST:")
    print(kq["test_table"].to_string(index=False))
