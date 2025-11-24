
# 📈 Stock Price Forecasting: Linear Family Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

> **Dự án đánh giá hiệu quả của các mô hình Linear, DLinear, và NLinear trong việc dự đoán giá cổ phiếu VIC (Vingroup) dựa trên dữ liệu lịch sử.**



[Image of Stock Market Chart generic]


## 📖 Mục tiêu (Objectives)
Dự án tập trung giải quyết bài toán dự báo Time-Series tài chính với dữ liệu cổ phiếu VIC từ **2020 - 2025**. Chúng tôi thực hiện benchmark so sánh hiệu năng giữa các biến thể của mô hình Linear trên các khung thời gian dự báo (Prediction Horizons) khác nhau:

* **Ngắn hạn:** 7 ngày, 30 ngày.
* **Trung & Dài hạn:** 120 ngày, 480 ngày.

---

## 📊 Dữ liệu & Tiền xử lý (Data Pipeline)

### 1. Nguồn dữ liệu
* **Dataset:** `VIC.csv`
* **Tổng quan:** ~1500 bản ghi.
* **Features:** `time`, `open`, `high`, `low`, `close`, `volume`, `symbol`.

### 2. Quy trình xử lý (Preprocessing)
Dữ liệu được đi qua một pipeline xử lý nghiêm ngặt để đảm bảo tính ổn định cho mô hình học sâu:

1.  **Cleaning:** Chuyển đổi `datetime`, sort theo thời gian, xử lý `NaN`.
2.  **Feature Engineering:**
    * `daily_return`: Biến động giá hàng ngày.
    * `close_log`: Logarit của giá đóng cửa ($\log(P)$) giúp chuỗi dữ liệu ổn định hơn, giảm thiểu tác động của biến động mạnh.
3.  **Normalization:** Sử dụng **StandardScaler** để đưa dữ liệu về phân phối chuẩn.
4.  **Splitting:** Chia tập dữ liệu theo thứ tự thời gian (No Shuffle):
    * 🟢 **Train:** 70%
    * 🟡 **Validation:** 15%
    * 🔴 **Test:** 15%

---

## 🧠 Kiến trúc Mô hình (Models)

Chúng tôi triển khai 3 biến thể hiện đại của mạng Linear cho Time Series:

| Mô hình | Mô tả | Đặc điểm |
| :--- | :--- | :--- |
| **Linear** | Single Layer Perceptron | Mapping trực tiếp từ `seq_len` $\rightarrow$ `pred_len`. Đơn giản nhưng hiệu quả. |
| **DLinear** | Decomposition Linear | Phân rã chuỗi thời gian thành **Trend** và **Remainder** trước khi đưa vào Linear layers. |
| **NLinear** | Normalization Linear | Trừ giá trị cuối cùng của input sequence để loại bỏ non-stationarity, sau đó cộng lại ở output. |

**Input Configuration:**
* **Look-back window (seq_len):** 7, 30, 120, 480 ngày.
* **Prediction horizon (pred_len):** 7 ngày.

---

## ⚙️ Huấn luyện (Training Config)

Quá trình huấn luyện được thực hiện độc lập cho từng cặp `Model` + `seq_len`.

* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Learning Rate:** `0.001`
* **Batch Size:** 32
* **Epochs:** 50

> 💾 **Checkpoint:** State_dict của mô hình có loss thấp nhất trên tập Val được lưu lại dưới dạng `.pth`.

---

## 🚀 Hướng dẫn sử dụng (Usage)

### 1. Load Pre-trained Model

```python
import torch
from models import Linear # Giả sử bạn để class model trong file models.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo architecture
seq_len = 30
pred_len = 7
model = Linear(seq_len, pred_len).to(device)

# Load weights
model.load_state_dict(torch.load("checkpoints/linear_30d_state_dict.pth", map_location=device))
model.eval()
````

### 2\. Dự báo giá (Inference)

```python
import numpy as np

# 1. Lấy dữ liệu input (seq_len ngày gần nhất) và chuẩn hóa
input_data = get_recent_data(days=30) 
x_tensor = torch.tensor(scaler.transform(input_data)).float().to(device)

# 2. Predict
with torch.no_grad():
    y_pred_log = model(x_tensor).cpu().numpy().flatten()

# 3. Inverse Scaling & Inverse Log để ra giá thực tế
y_pred_denorm = scaler.inverse_transform(y_pred_log.reshape(-1,1)).flatten()
predicted_price = np.exp(y_pred_denorm)

print(f"Dự đoán giá VIC 7 ngày tới: {predicted_price}")
```

-----

## 📉 Đánh giá (Evaluation)

Hiệu năng mô hình được đo lường trên tập Test bằng các metrics tiêu chuẩn:

  * **MSE** (Mean Squared Error)
  * **MAE** (Mean Absolute Error)
  * **RMSE** (Root Mean Squared Error)
  * **$R^2$ Score**

### Kết quả sơ bộ

  * Hiệu quả dự đoán có xu hướng **giảm** khi `seq_len` tăng quá lớn (do nhiễu dữ liệu lịch sử xa).
  * **DLinear** và **NLinear** thường cho kết quả ổn định hơn Linear thuần túy trong các giai đoạn thị trường biến động mạnh (trend changes).

-----

## 📝 Ghi chú (Notes)

1.  ⚠️ **Dimension Mismatch:** Đảm bảo `seq_len` khi khởi tạo mô hình khớp chính xác với file `state_dict` đã lưu.
2.  📈 **Log Transformation:** Output của mô hình là `log_price`. Đừng quên dùng hàm `exp()` để chuyển về giá VND thực tế.

-----

### Author

Developed for Stock Prediction Research.

