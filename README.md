# Project 5: Machine Learning (CS188) - Nhóm 5 - INT3401_11

Bài tập lớn về học máy với **PyTorch**, gồm 7 câu: Perceptron, hồi quy phi tuyến, phân loại chữ số, nhận dạng ngôn ngữ bằng RNN, CNN, Attention và character-GPT.

- Đề bài gốc: <https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/>
- Hạn nộp ghi trên đề gốc: Thứ Sáu 21/11, 23:59 (giờ PT). Hạn thực tế của lớp theo thông báo của giảng viên.

## Mục lục

1. [Tổng quan và điểm số](#1-tổng-quan-và-điểm-số)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Cách chạy và chấm thử](#4-cách-chạy-và-chấm-thử)
5. [Mô tả từng câu](#5-mô-tả-từng-câu)
6. [Kết quả](#6-kết-quả)
7. [Nộp bài](#7-nộp-bài)
8. [Lỗi thường gặp](#8-lỗi-thường-gặp)
9. [Phân công nhóm](#9-phân-công-nhóm)

---

## 1. Tổng quan và điểm số

| Câu | Nội dung | Điểm | Ngưỡng đạt (theo đề) |
|---|---|---|---|
| Q1 | Perceptron nhị phân | 5 | Đạt 100% accuracy trên tập huấn luyện |
| Q2 | Hồi quy phi tuyến, xấp xỉ sin(x) | 5 | Loss ≤ 0.02 |
| Q3 | Phân loại chữ số MNIST | 5 | Test accuracy ≥ 97% |
| Q4 | Nhận dạng ngôn ngữ (RNN) | 5 | Test accuracy ≥ 81% |
| Q5 | Mạng tích chập (CNN) | 3 | Convolve đúng và test accuracy ≥ 80% |
| Q6 | Attention | 2 | Đúng kết quả so với autograder |
| Q7 | character-GPT (tùy chọn) | 0 | Không chấm, làm để tìm hiểu |
| | **Tổng** | **25** | |

## 2. Cấu trúc thư mục

| File / thư mục | Nội dung | Có được sửa? |
|---|---|---|
| `models.py` | Các mô hình: `PerceptronModel`, `RegressionModel`, `DigitClassificationModel`, `LanguageIDModel`, `Convolve`, `DigitConvolutionalModel`, `Attention` | Có |
| `train.py` | Vòng huấn luyện cho từng mô hình | Có |
| `losses.py` | Hàm loss cho từng bài | Có |
| `gpt_model.py` | `Transformer_Block` và `Character_GPT` (Q7) | Có (chỉ Q7) |
| `chargpt.py` | Huấn luyện và sinh văn bản cho Q7 | Không |
| `input.txt` | Văn bản Shakespeare dùng cho Q7 (tải từ trang đề) | Không |
| `autograder.py`, `backend.py`, `data/` | Do giáo viên cung cấp | **Không** |

Quy tắc quan trọng: không đổi tên hàm hoặc lớp có sẵn, và không thêm hoặc sửa phần `import` ở đầu file, nếu không autograder sẽ không chấm được.

## 3. Cài đặt môi trường

Yêu cầu:

- Python **3.11** (khuyến nghị). Python 3.14 quá mới nên không cài được `numpy==1.24.3`.
- Thư viện: `numpy==1.24.3` (hoặc bản < 2.0), `matplotlib`, `torch` (bản CPU).

Cài đặt trên Windows (PowerShell):

```powershell
# 1. Vào thư mục dự án
cd <đường_dẫn_tới_thư_mục_machinelearning>

# 2. Tạo môi trường ảo (nên đặt ngoài OneDrive để tránh đồng bộ chậm)
py -3.11 -m venv C:\venv_proj5

# 3. Bật môi trường ảo
C:\venv_proj5\Scripts\activate

# 4. Cài thư viện
pip install numpy==1.24.3 matplotlib torch

# 5. Kiểm tra: hiện cửa sổ có đoạn thẳng xoay vòng tròn là thành công
python autograder.py --check-dependencies
```

Nếu PowerShell báo `running scripts is disabled`, chạy một lần:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Trên macOS hoặc Linux: thay `py -3.11` bằng `python3.11`, và bật môi trường bằng `source <thư_mục_venv>/bin/activate`.

Lưu ý: mỗi lần mở terminal mới, cần `cd` vào thư mục dự án và bật lại môi trường ảo trước khi chạy.

## 4. Cách chạy và chấm thử

Chấm từng câu (thêm `--no-graphics` để không mở cửa sổ đồ họa):

```powershell
python autograder.py -q q1 --no-graphics   # Perceptron
python autograder.py -q q2 --no-graphics   # Hồi quy phi tuyến
python autograder.py -q q3 --no-graphics   # Phân loại chữ số (MNIST)
python autograder.py -q q4 --no-graphics   # Nhận dạng ngôn ngữ (RNN)
python autograder.py -q q5 --no-graphics   # CNN
python autograder.py -q q6 --no-graphics   # Attention
```

Chấm toàn bộ (khoảng 12 phút trở lên, tùy máy):

```powershell
python autograder.py --no-graphics
```

Chạy Q7 (không có autograder, cần `input.txt` cùng thư mục):

```powershell
python chargpt.py
```

Cách đọc kết quả: `PASS` kèm `Question qN: x/x` là đạt. Điểm autograder ở máy chỉ là **điểm tạm tính**, điểm chính thức do Gradescope và giảng viên quyết định.

Mỗi câu có yếu tố ngẫu nhiên (khởi tạo trọng số, xáo trộn dữ liệu). Nếu một câu rớt sát ngưỡng, chạy lại. Nếu rớt hai lần liên tiếp thì chỉnh siêu tham số.

## 5. Mô tả từng câu

### Q1: Perceptron nhị phân

- **Mô hình:** một vector trọng số `w` kiểu `Parameter`, shape `1 × dimensions`. `forward` trả về tích vô hướng `w·x`. `get_prediction` trả về `+1` nếu điểm số ≥ 0, ngược lại `-1`.
- **Huấn luyện:** duyệt từng mẫu (batch size 1). Nếu dự đoán sai thì cập nhật `w += y·x`. Dừng khi một lượt đầy đủ không còn mẫu sai (100% accuracy trên tập huấn luyện).

### Q2: Hồi quy phi tuyến (xấp xỉ sin(x) trên [-2π, 2π])

- **Mô hình:** `Linear(1→128)` → ReLU → `Linear(128→128)` → ReLU → `Linear(128→1)`.
- **Loss:** `mse_loss`.
- **Huấn luyện:** Adam (lr 0.005), batch 25, dừng khi loss huấn luyện đủ nhỏ (ngưỡng đề: ≤ 0.02).

### Q3: Phân loại chữ số viết tay (MNIST)

- **Đầu vào:** ảnh 28×28 trải phẳng thành 784 số. **Đầu ra:** 10 điểm số cho các chữ số 0 đến 9.
- **Mô hình:** `Linear(784→256)` → ReLU → `Linear(256→128)` → ReLU → `Linear(128→10)`. Lớp cuối không có ReLU.
- **Loss:** `cross_entropy`.
- **Huấn luyện:** Adam (lr 0.001) giảm dần theo `StepLR`, batch 100, dừng sớm theo validation accuracy.
- Đề lưu ý: autograder chấm trên tập test, còn ta chỉ thấy validation. Nên đặt ngưỡng dừng cao hơn một chút (khoảng 97.5% đến 98%) để có biên an toàn.

### Q4: Nhận dạng ngôn ngữ bằng RNN

- **Bài toán:** cho một từ, đoán nó thuộc 1 trong 5 ngôn ngữ (English, Spanish, Finnish, Dutch, Polish). Mỗi ký tự là vector one-hot 47 chiều. Từ có độ dài khác nhau nên dùng RNN.
- **Mô hình:** trạng thái ẩn `h` có 256 chiều, gồm 3 lớp `Linear`: `Wx` (47→256), `Wh` (256→256), `out` (256→5).
  - Ký tự đầu: `h = relu(Wx·x₀)`
  - Các ký tự sau: `h = relu(Wx·xᵢ + Wh·h)`
  - Cuối từ: `scores = out(h)`, không có ReLU.
  - `Wx` và `Wh` được dùng chung ở mọi bước nên xử lý được từ có độ dài bất kỳ.
- **Loss:** `cross_entropy`.
- **Huấn luyện:** Adam (lr 0.002), batch 64, tối đa 30 epoch, dừng sớm khi accuracy ≥ 86%. Mỗi batch từ DataLoader có shape `(batch, độ dài, 47)`, được đổi bằng `movedim` thành `(độ dài, batch, 47)` để `forward` duyệt theo từng ký tự.

### Q5: Mạng tích chập (CNN)

- **`Convolve(input, weight)`:** tự cài phép tích chập 2D, không đệm 0, nên ảnh 28×28 với kernel 3×3 cho ra 26×26. Cách làm: cộng các lát cắt dịch chuyển của ảnh, mỗi lát nhân với một phần tử của kernel.
- **Mô hình:** lớp tích chập (có sẵn trong khung) rồi trải phẳng → `Linear(676→128)` → ReLU → `Linear(128→10)`.
- **Loss:** `cross_entropy`.
- **Huấn luyện:** Adam (lr 0.002), batch 50, dừng sớm theo validation accuracy. Bài này dùng một tập MNIST rút gọn nên huấn luyện khá nhanh.

### Q6: Attention (Scaled Dot-Product, có mặt nạ nhân quả)

- **Công thức:** `softmax( mask( Q·Kᵀ / √d ) ) · V`. Trong đó `Q`, `K`, `V` được tạo từ cùng một đầu vào qua ba lớp `Linear` riêng (`q_layer`, `k_layer`, `v_layer`), `d` là `layer_size`.
- **Mặt nạ nhân quả:** ký tự ở vị trí `i` chỉ được nhìn các vị trí `0…i`. Các ô bị che được gán `-inf` bằng `masked_fill` trước khi `softmax` theo chiều cuối.
- Phép chuyển vị `Kᵀ` thực hiện bằng `movedim(K, 1, 2)`.

### Q7: character-GPT (tùy chọn, 0 điểm)

Mô hình ngôn ngữ mức ký tự, huấn luyện trên văn bản Shakespeare (`input.txt`), dự đoán ký tự tiếp theo rồi sinh văn bản từ prompt `"Pacman"`.

- **`Transformer_Block`:**
  ```
  x = LayerNorm_1( x + Attention(x) )
  x = LayerNorm_2( x + ReLU(Linear(x)) )
  ```
- **`Character_GPT.forward`:** `Embedding` → `n_layer` khối Transformer nối tiếp → `LayerNorm` → `Linear` (không bias, không activation) → logits.
- **Cấu hình:** `block_size` = 10, `layer_size` = 100, `n_layer` = 6, `batch_size` = 500, 10 000 vòng lặp, optimizer Adam, loss cross-entropy.
- **Kết quả:** val loss giảm từ 4.15 (vòng 0) xuống khoảng 1.82 đến 1.83 (vòng 9800 đến 9900). Văn bản sinh ra học được định dạng kịch bản (tên nhân vật viết hoa kèm dấu hai chấm) và nhiều từ tiếng Anh cổ, nhưng câu chưa có nghĩa vì ngữ cảnh chỉ có 10 ký tự.
- Chi tiết xem `README_Q7.md`.

## 6. Kết quả

Số liệu ghi theo các lần chạy của nhóm (có thể dao động nhẹ giữa các lần chạy do tính ngẫu nhiên).

| Câu | Ngưỡng đạt | Kết quả | Điểm |
|---|---|---|---|
| Q1 | 100% accuracy huấn luyện | Pass | 5/5 |
| Q2 | Loss ≤ 0.02 | Loss ≈ 0.0092 | 5/5 |
| Q3 | Test accuracy ≥ 97% | ≈ 97.96% | 5/5 |
| Q4 | Test accuracy ≥ 81% | 86.2% (dừng ở epoch 17) | 5/5 |
| Q5 | Test accuracy ≥ 80% | ≈ 92% | 3/3 |
| Q6 | Khớp kết quả autograder | Pass | 2/2 |
| Q7 | Không chấm | Val loss ≈ 1.83 | 0 |
| | **Tổng** | | **25/25** |

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `python` hiện Python 3.14 thay vì 3.11 | Chưa bật môi trường ảo, chạy lại `C:\venv_proj5\Scripts\activate` |
| `git is not recognized` | Máy chưa cài Git, cài từ git-scm.com rồi mở lại terminal |
| `IndentationError` | Thụt lề lệch khi dán code, mỗi cấp thụt 4 dấu cách |
| `AttributeError ... get_weights` (Q1) | Thiếu hàm `get_weights` trong `PerceptronModel` |
| Loss ra `nan` hoặc `inf` | Learning rate quá lớn, giảm một nửa |
| Lỗi numpy khi import | Cài lại bản cũ: `pip install numpy==1.24.3` |
| Test accuracy rớt sát ngưỡng | Chạy lại, hoặc nâng ngưỡng dừng sớm trên validation |
| Q5 chạy chậm | Bình thường vì `Convolve` tự cài, bài dùng tập MNIST rút gọn nên vẫn đủ nhanh |
| Q7 báo không tìm thấy `input.txt` | Tải file từ trang đề và đặt vào thư mục dự án |

