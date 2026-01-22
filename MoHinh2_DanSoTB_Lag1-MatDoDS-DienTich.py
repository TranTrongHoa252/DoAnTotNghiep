# =====================================================
# HỒI QUY ĐA BIẾN 
# Sử dụng: DanSoTB_lag1, MatDoDS, DienTich
# Dự báo: DanSoTB
# =====================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Optional, Dict

FEATURES_DEFAULT: List[str] = ["DanSoTB_lag1", "MatDoDS", "DienTich"]

# 1. LOAD + TIỀN XỬ LÝ DỮ LIỆU
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df["Nam"] = df["Nam"].astype(int)

    # Quy đổi dân số trung bình: nghìn người → người
    df["DanSoTB"] = df["DanSoTB"] * 1000
    return df

# 2. HÀM DỰ BÁO ĐA BIẾN
def du_bao_da_bien_lag1_matdo_dientich(
    df: pd.DataFrame,
    tinh: str,
    nam_chon: int,
    test_years: int = 2,
    features: Optional[List[str]] = None,
    x_nam_du_bao: Optional[Dict[str, float]] = None
):
    if features is None:
        features = FEATURES_DEFAULT

    # Lọc dữ liệu theo tỉnh
    g = df[df["Tinh"] == tinh].sort_values("Nam").copy()
    if g.empty:
        raise ValueError(f"Không tìm thấy tỉnh: {tinh}")

    #Tạo biến trễ(lag1 - là năm trước năm chọn dự báo )
    g["DanSoTB_lag1"] = g["DanSoTB"].shift(1)
    g = g.dropna().copy()

    #Chỉ dùng dữ liệu quá khứ
    g = g[g["Nam"] < nam_chon].copy()
    if len(g) < test_years + 2:
        raise ValueError("Không đủ dữ liệu để train/test")

    # Chia train / test theo thời gian
    train = g.iloc[:-test_years]
    test = g.iloc[-test_years:]

    X_train = train[features]
    y_train = train["DanSoTB"]

    X_test = test[features]
    y_test = test["DanSoTB"]

    # 3. CHUẨN HÓA 
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # 4. HUẤN LUYỆN MÔ HÌNH
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    # 5. ĐÁNH GIÁ MÔ HÌNH
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    ).ravel()

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred) if len(test) >= 2 else np.nan

    test = test.copy()
    test["DanSoDuBao"] = y_test_pred

    # 6. DỰ BÁO NĂM CHỌN
    nam_truoc = nam_chon - 1
    if nam_truoc not in g["Nam"].values:
        raise ValueError(f"Không có dữ liệu năm {nam_truoc}")

    dan_so_lag1 = g.loc[g["Nam"] == nam_truoc, "DanSoTB"].iloc[0]

    row_future = df[(df["Tinh"] == tinh) & (df["Nam"] == nam_chon)]
    if not row_future.empty:
        matdo = row_future["MatDoDS"].iloc[0]
        dientich = row_future["DienTich"].iloc[0]
        source_inputs = "from_file"
    else:
        if x_nam_du_bao is None:
            raise ValueError("Thiếu MatDoDS và DienTich cho năm dự báo")
        matdo = x_nam_du_bao["MatDoDS"]
        dientich = x_nam_du_bao["DienTich"]
        source_inputs = "from_user"

    X_forecast = pd.DataFrame([{
        "DanSoTB_lag1": dan_so_lag1,
        "MatDoDS": matdo,
        "DienTich": dientich
    }])

    X_forecast_scaled = scaler_X.transform(X_forecast)
    du_bao_scaled = model.predict(X_forecast_scaled)

    du_bao = scaler_y.inverse_transform(
        du_bao_scaled.reshape(-1, 1)
    )[0][0]

    # 7. TỔNG HỢP KẾT QUẢ
    coef_lines = "\n".join(
        [f"  - {features[i]}: {model.coef_[i]:.6f}" for i in range(len(features))]
    )

    summary_text = (
        f"Tỉnh: {tinh}\n"
        f"Năm dự báo: {nam_chon}\n"
        f"Mô hình: DanSoTB_t = b0 + b1*DanSoTB_(t-1) + b2*MatDoDS + b3*DienTich\n"
        f"Hệ số:\n{coef_lines}\n"
        f"Đánh giá mô hình:\n"
        f"  - MAE : {mae:,.0f}\n"
        f"  - RMSE: {rmse:,.0f}\n"
        f"  - R2  : {r2:.4f}\n"
        f"Dự báo dân số năm {nam_chon}: {du_bao:.0f} người(Nguồn: {source_inputs})\n"
    )

    return {
        "summary_text": summary_text,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "test_table": test[["Nam"] + features + ["DanSoTB", "DanSoDuBao"]],
        "forecast": du_bao
    }

# 8. CHẠY THỬ
if __name__ == "__main__":
    df = load_and_preprocess("DataDanSoVN.csv")

    kq = du_bao_da_bien_lag1_matdo_dientich(
        df,
        tinh="HaNoi",
        nam_chon=2023,
        test_years=2
    )

    print(kq["summary_text"])
    print("BẢNG TEST 2 NĂM TRƯỚC CỦA NĂM ĐƯỢC CHỌN")
    print(kq["test_table"].to_string(index=False))
