# Project 5: Machine Learning (CS188 Fall 2025)

Bài tập lớn về học máy với PyTorch: perceptron, hồi quy, phân loại chữ số, nhận dạng ngôn ngữ bằng RNN, CNN và Attention.

Đề bài: https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/

## 1. Yêu cầu hệ thống

- Python **3.11** (khuyến nghị; Python 3.14 quá mới nên không cài được `numpy==1.24.3`)
- Thư viện: `numpy==1.24.3`, `matplotlib`, `torch` (bản CPU)

## 2. Cài đặt (Windows / PowerShell)

```powershell
# 1. Vào thư mục dự án
cd <đường_dẫn_tới_thư_mục_machinelearning>

# 2. Tạo môi trường ảo (nên đặt ngoài OneDrive để tránh đồng bộ chậm)
py -3.11 -m venv C:\venv_proj5

# 3. Bật môi trường ảo
C:\venv_proj5\Scripts\activate

# 4. Cài thư viện
pip install numpy==1.24.3 matplotlib torch

# 5. Kiểm tra cài đặt (hiện cửa sổ có đoạn thẳng xoay vòng tròn là thành công)
python autograder.py --check-dependencies
```

Nếu PowerShell báo `running scripts is disabled`, chạy một lần:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Với macOS/Linux, thay `py -3.11` bằng `python3.11` và bật môi trường bằng `source <thư_mục_venv>/bin/activate`.

**Lưu ý:** mỗi lần mở terminal mới, cần `cd` vào thư mục dự án và bật lại môi trường ảo trước khi chạy.

## 3. Chạy chương trình

Chạy từng câu (thêm `--no-graphics` để chạy không cần cửa sổ đồ họa):

```powershell
python autograder.py -q q1 --no-graphics   # Perceptron
python autograder.py -q q2 --no-graphics   # Hồi quy phi tuyến
python autograder.py -q q3 --no-graphics   # Phân loại chữ số (MNIST)
python autograder.py -q q4 --no-graphics   # Nhận dạng ngôn ngữ (RNN)
python autograder.py -q q5 --no-graphics   # CNN
python autograder.py -q q6 --no-graphics   # Attention
```

Chạy toàn bộ (mất khoảng 12 phút trở lên tùy máy):

```powershell
python autograder.py --no-graphics
```

Mỗi câu có yếu tố ngẫu nhiên (khởi tạo trọng số, xáo trộn dữ liệu). Nếu một câu rớt sát ngưỡng, chạy lại; nếu rớt hai lần liên tiếp thì chỉnh siêu tham số.

## 4. Cấu trúc file

| File | Nội dung |
|---|---|
| `models.py` | Các mô hình: `PerceptronModel`, `RegressionModel`, `DigitClassificationModel`, `LanguageIDModel`, `Convolve` + `DigitConvolutionalModel`, `Attention` |
| `train.py` | Vòng huấn luyện cho từng mô hình |
| `losses.py` | Hàm loss cho từng bài |
| `autograder.py`, `backend.py`, `data/` | Do giáo viên cung cấp, **không chỉnh sửa** |
| `chargpt.py`, `gpt_model.py` | Q7 (character-GPT, tùy chọn, 0 điểm, chưa thực hiện) |

Chỉ nộp `models.py`, `train.py`, `losses.py`. Không đổi tên hàm/lớp có sẵn và không thêm import ngoài danh sách cho phép, nếu không autograder sẽ không chấm được.

## 5. Mô tả từng câu

### Q1: Perceptron nhị phân
- **Mô hình:** một vector trọng số `w`; điểm số là tích vô hướng `w·x`, dự đoán `+1` nếu ≥ 0, ngược lại `-1`.
- **Huấn luyện:** lặp qua dữ liệu, mỗi lần dự đoán sai thì cập nhật `w += y·x`; dừng khi một lượt không còn mẫu sai.

### Q2: Hồi quy phi tuyến (xấp xỉ sin(x) trên [-2π, 2π])
- **Mô hình:** mạng Linear(1→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1).
- **Huấn luyện:** loss `mse_loss`, Adam (lr 0.005), batch 25, dừng khi loss huấn luyện đủ nhỏ (ngưỡng đề: ≤ 0.02).

### Q3: Phân loại chữ số viết tay (MNIST)
- **Mô hình:** ảnh 28×28 trải phẳng thành 784 số; mạng Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10), lớp cuối không có ReLU.
- **Huấn luyện:** loss `cross_entropy`, Adam (lr 0.001) giảm dần theo `StepLR`, batch 100, dừng sớm theo validation accuracy.

### Q4: Nhận dạng ngôn ngữ bằng RNN
- **Bài toán:** cho một từ, dự đoán 1 trong 5 ngôn ngữ (English, Spanish, Finnish, Dutch, Polish); mỗi ký tự là vector one-hot 47 chiều.
- **Mô hình:** RNN với trạng thái ẩn 256 chiều. Ký tự đầu: `h = relu(Wx·x₀)`; các ký tự sau: `h = relu(Wx·xᵢ + Wh·h)`; cuối cùng `scores = Linear(h)` (5 đầu ra, không ReLU). `Wx`, `Wh` dùng chung ở mọi bước nên xử lý được từ có độ dài bất kỳ.
- **Huấn luyện:** loss `cross_entropy`, Adam (lr 0.002), batch 64, tối đa 30 epoch, dừng sớm khi validation accuracy ≥ 86%. Batch được đổi shape bằng `movedim` từ `(batch, độ dài, 47)` sang `(độ dài, batch, 47)`.

### Q5: Mạng tích chập (CNN)
- **`Convolve`:** tự cài phép tích chập 2D không đệm 0 (ảnh 28×28, kernel 3×3 cho ra 26×26), bằng cách cộng các lát cắt dịch chuyển của ảnh nhân với từng phần tử kernel.
- **Mô hình:** convolution + trải phẳng (có sẵn trong khung) → Linear(676→128) → ReLU → Linear(128→10).
- **Huấn luyện:** loss `cross_entropy`, Adam (lr 0.002), batch 50, dừng sớm theo validation accuracy.

### Q6: Attention
- **Công thức:** `softmax(mask(Q·Kᵀ / √d)) · V`, với Q, K, V sinh từ cùng đầu vào qua các lớp Linear riêng (`q_layer`, `k_layer`, `v_layer`).
- **Mặt nạ nhân quả:** ký tự ở vị trí `i` chỉ nhìn được các vị trí `0…i`; các ô bị che gán `-inf` trước softmax (dùng `masked_fill`).

## 6. Kết quả

| Câu | Nội dung | Ngưỡng đạt | Kết quả | Điểm |
|---|---|---|---|---|
| Q1 | Perceptron | Đạt 100% accuracy huấn luyện | Pass | 5/5 |
| Q2 | Hồi quy xấp xỉ sin(x) | Loss ≤ 0.02 | Loss ≈ 0.0092 | 5/5 |
| Q3 | Phân loại chữ số | Test accuracy ≥ 97% | ≈ 97.96% | 5/5 |
| Q4 | Nhận dạng ngôn ngữ | Test accuracy ≥ 81% | ≈ 86.2% | 5/5 |
| Q5 | CNN | Test accuracy ≥ 80% | ≈ 92% | 3/3 |
| Q6 | Attention | Pass autograder | Pass | 2/2 |
| | **Tổng** | | | **25/25** |

(Con số cụ thể có thể dao động nhẹ giữa các lần chạy do tính ngẫu nhiên.)

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `python` hiện Python 3.14 thay vì 3.11 | Chưa bật môi trường ảo; chạy lại `C:\venv_proj5\Scripts\activate` |
| `IndentationError` | Dòng code mới thụt lề lệch; mỗi cấp thụt 4 dấu cách |
| `AttributeError ... get_weights` (Q1) | Thiếu hàm `get_weights` trong `PerceptronModel` |
| Loss ra `nan` | Learning rate quá lớn; giảm xuống một nửa |
| Lỗi numpy khi import | Cài lại bản cũ: `pip install numpy==1.24.3` |
| Test accuracy rớt sát ngưỡng | Chạy lại, hoặc nâng ngưỡng dừng sớm trên validation |
