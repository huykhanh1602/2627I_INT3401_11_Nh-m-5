# Project 5: Machine Learning (CS188) - Nhóm 5 - INT3401_11

Bài tập lớn về học máy với **PyTorch**, gồm 7 câu: Perceptron, hồi quy phi tuyến, phân loại chữ số, nhận dạng ngôn ngữ bằng RNN, CNN, Attention và character-GPT.

- Đề bài gốc: <https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/>

## Mục lục

1. [Tổng quan và điểm số](#1-tổng-quan-và-điểm-số)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Cách chạy và chấm thử](#4-cách-chạy-và-chấm-thử)
5. [Mô tả từng câu](#5-mô-tả-từng-câu)
6. [Kết quả](#6-kết-quả)
---

## 1. Tổng quan và điểm số

| Câu | Nội dung | Điểm | Ngưỡng đạt (theo đề) |
|---|---|---|---|
| Q1 | Perceptron nhị phân | 5 | Đạt 100% accuracy trên tập huấn luyện |
| Q2 | Hồi quy phi tuyến, xấp xỉ sin(x) | 5 | Loss ≤ 0.02 |
| Q3 | Phân loại chữ số MNIST | 5 | Test accuracy ≥ 97% |
| Q4 | Nhận dạng ngôn ngữ (RNN) | 5 | Test accuracy ≥ 81% |
| Q5 | Mạng tích chập (CNN) | 3 | `Convolve` đúng và test accuracy ≥ 80% |
| Q6 | Attention | 2 | Đúng kết quả so với autograder |
| Q7 | character-GPT (tùy chọn) | 0 | Không chấm, làm để tìm hiểu |
| | **Tổng** | **25** | |

## 2. Cấu trúc thư mục

| File / thư mục | Nội dung | Có được sửa? |
|---|---|---|
| `models.py` | Các mô hình: `PerceptronModel`, `RegressionModel`, `DigitClassificationModel`, `LanguageIDModel`, `Convolve`, `DigitConvolutionalModel`, `Attention` | Có |
| `train.py` | Vòng huấn luyện cho từng mô hình | Có |
| `losses.py` | Hàm loss cho từng bài | Có |
| `gpt_model.py` | `Transformer_Block` và `Character_GPT` (Q7) | Đã có sẵn code ở các chỗ cần điền khi nhóm nhận file |
| `chargpt.py` | Huấn luyện và sinh văn bản cho Q7 | Có, chỉnh để đo val loss (xem mục Q7). Bản gốc lưu ở `chargpt_goc.py` |
| `input.txt` | Văn bản Shakespeare cho Q7 (tải từ trang đề) | Không |
| `demo.py` | Giao diện demo (Tkinter), mỗi câu một tab | File của nhóm, không nộp chấm |
| `ui_weights/` | Trọng số do `demo.py` lưu sau khi huấn luyện | Tự sinh ra |
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

Chạy từng câu (thêm `--no-graphics` để không mở cửa sổ đồ họa):

```powershell
python autograder.py -q q1 --no-graphics   # Perceptron
python autograder.py -q q2 --no-graphics   # Hồi quy phi tuyến
python autograder.py -q q3 --no-graphics   # Phân loại chữ số (MNIST)
python autograder.py -q q4 --no-graphics   # Nhận dạng ngôn ngữ (RNN)
python autograder.py -q q5 --no-graphics   # CNN
python autograder.py -q q6 --no-graphics   # Attention
```

Chạy toàn bộ (khoảng 12 phút trở lên, tùy máy):

```powershell
python autograder.py --no-graphics
```

Chạy Q7 (không có autograder, cần `input.txt` cùng thư mục). Dùng `-u` để chữ hiện ngay khi ghi log:

```powershell
python -u chargpt.py | Tee-Object logQ7.txt
```

Chạy demo giao diện (cần thư mục `data/` và, cho Q7, `input.txt`):

```powershell
python demo.py
```

Mỗi tab của demo có nút Train và nút "Nạp trọng số đã lưu". Nên huấn luyện trước từng tab để trọng số được lưu vào `ui_weights/`, lúc trình bày chỉ cần nạp lại. Mỗi lúc chỉ chạy được một tác vụ huấn luyện.

## 5. Mô tả từng câu

### Q1: Perceptron nhị phân

- **Mô hình:** một vector trọng số `w` kiểu `Parameter`, shape `1 × dimensions`. `forward` trả về tích vô hướng `w·x`. `get_prediction` trả về `+1` nếu điểm số ≥ 0, ngược lại `-1`. `get_weights` trả về `w` cho autograder.
- **Huấn luyện:** duyệt từng mẫu (batch size 1). Nếu dự đoán sai thì cập nhật `w += y·x`. Dừng khi một lượt đầy đủ không còn mẫu sai (100% accuracy trên tập huấn luyện).

### Q2: Hồi quy phi tuyến (xấp xỉ sin(x) trên [-2π, 2π])

- **Mô hình:** `Linear(1→128)` → ReLU → `Linear(128→128)` → ReLU → `Linear(128→1)`.
- **Loss:** `mse_loss`.
- **Huấn luyện:** Adam (lr 0.005), batch 25, dừng khi loss huấn luyện dưới 0.01 (ngưỡng của đề là 0.02).

### Q3: Phân loại chữ số viết tay (MNIST)

- **Đầu vào:** ảnh 28×28 trải phẳng thành 784 số. **Đầu ra:** 10 điểm số cho các chữ số 0 đến 9.
- **Mô hình:** `Linear(784→400)` → ReLU → `Linear(400→200)` → ReLU → `Linear(200→10)`. Lớp cuối không có ReLU.
- **Loss:** `cross_entropy`.
- **Huấn luyện:** Adam (lr 0.001) giảm một nửa sau mỗi 3 epoch (`StepLR`), batch 100, tối đa 30 epoch, dừng sớm khi validation accuracy ≥ 98.2%.
- Đề lưu ý: autograder chấm trên tập test, còn ta chỉ thấy validation, nên đặt ngưỡng dừng cao hơn ngưỡng đề (97%) để có biên an toàn.

### Q4: Nhận dạng ngôn ngữ bằng RNN

- **Bài toán:** cho một từ, đoán nó thuộc 1 trong 5 ngôn ngữ (English, Spanish, Finnish, Dutch, Polish). Mỗi ký tự là vector one-hot 47 chiều. Từ có độ dài khác nhau nên dùng RNN.
- **Mô hình:** trạng thái ẩn `h` có 256 chiều, gồm 3 lớp `Linear`: `Wx` (47→256), `Wh` (256→256), `out` (256→5).
  - Ký tự đầu: `h = relu(Wx·x₀)`
  - Các ký tự sau: `h = relu(Wx·xᵢ + Wh·h)`
  - Cuối từ: `scores = out(h)`, không có ReLU.
  - `Wx` và `Wh` được dùng chung ở mọi bước nên xử lý được từ có độ dài bất kỳ.
- **Loss:** `cross_entropy`.
- **Huấn luyện:** Adam (lr 0.002), batch 64, tối đa 30 epoch, dừng sớm khi validation accuracy ≥ 86%. Mỗi batch từ DataLoader có shape `(batch, độ dài, 47)`, được đổi bằng `movedim` thành `(độ dài, batch, 47)` để `forward` duyệt theo từng ký tự.

### Q5: Mạng tích chập (CNN)

- **`Convolve(input, weight)`:** tự cài phép tích chập 2D, không đệm 0, nên ảnh 28×28 với kernel 3×3 cho ra 26×26. Cách làm: cộng các lát cắt dịch chuyển của ảnh, mỗi lát nhân với một phần tử của kernel.
- **Mô hình:** lớp tích chập (có sẵn trong khung) rồi trải phẳng → `Linear(676→128)` → ReLU → `Linear(128→10)`.
- **Loss:** `cross_entropy`.
- **Huấn luyện:** Adam (lr 0.002), batch 50, tối đa 15 epoch, dừng sớm khi validation accuracy ≥ 86%. Bài này dùng một tập MNIST rút gọn nên huấn luyện khá nhanh.

### Q6: Attention (Scaled Dot-Product, có mặt nạ nhân quả)

- **Công thức:** `softmax( mask( Q·Kᵀ / √d ) ) · V`. Trong đó `Q`, `K`, `V` được tạo từ cùng một đầu vào qua ba lớp `Linear` riêng (`q_layer`, `k_layer`, `v_layer`), `d` là `layer_size`.
- **Mặt nạ nhân quả:** ký tự ở vị trí `i` chỉ được nhìn các vị trí `0…i`. Các ô bị che được gán `-inf` bằng `masked_fill` trước khi `softmax` theo chiều cuối.
- Phép chuyển vị `Kᵀ` thực hiện bằng `movedim(K, 1, 2)`.

### Q7: character-GPT (tùy chọn, 0 điểm)

Mô hình ngôn ngữ mức ký tự, huấn luyện trên văn bản Shakespeare (`input.txt`), dự đoán ký tự tiếp theo rồi sinh văn bản từ prompt `"Pacman"`.

**Kiến trúc**

- **`Transformer_Block`:**
  ```
  x = LayerNorm_1( x + Attention(x) )
  x = LayerNorm_2( x + ReLU(Linear(x)) )
  ```
  Hai phép cộng `x + ...` là kết nối tắt (residual), giúp gradient truyền qua nhiều khối.
- **`Character_GPT.forward`:** `Embedding` → `n_layer` khối Transformer nối tiếp → `LayerNorm` → `Linear` (không bias, không activation) → logits.
- Code trong `Transformer_Block.forward` và `Character_GPT.forward` đã có sẵn khi nhóm nhận file `gpt_model.py`. Phần việc của nhóm ở Q7 là chạy huấn luyện, đo val loss và đánh giá kết quả. Q7 phụ thuộc vào lớp `Attention` của Q6.

**Thiết lập thí nghiệm**

| Tham số | Giá trị |
|---|---|
| `block_size` (độ dài ngữ cảnh) | 10 |
| `layer_size` (số chiều embedding) | 100 |
| `n_layer` (số khối Transformer) | 6 |
| `batch_size` | 500 |
| Số vòng lặp | 10 000 |
| Optimizer | Adam, learning rate **0.0004** |
| Loss | Cross-entropy |
| Dữ liệu | 90% đầu của `input.txt` để huấn luyện, 10% cuối để đo val loss |
| Seed | 0 |

**Thay đổi so với `chargpt.py` gốc của giảng viên** (bản gốc lưu ở `chargpt_goc.py`):

- Thêm hàm `eval_loss` đo val loss (trung bình 20 batch ngẫu nhiên trên 10% dữ liệu không dùng để huấn luyện), và in train loss cùng val loss mỗi 100 vòng.
- Đặt seed để chạy lại ra kết quả tương tự.
- Bản gốc khai báo biến `learning_rate = 0.0004` nhưng dòng tạo optimizer lại ghi cứng `lr=0.00004` nên biến đó không có tác dụng. Bản của nhóm dùng `lr=learning_rate`, tức là lr thực tế lớn hơn bản gốc 10 lần.

**Kết quả (log huấn luyện của nhóm)**

| Vòng lặp | Train loss | Val loss |
|---|---|---|
| 0 | 4.2876 | 4.1452 |
| 100 | 2.4544 | 2.4801 |
| 1000 | 1.9705 | 2.0517 |
| 2000 | 1.8586 | 1.9751 |
| 5000 | 1.7227 | 1.8902 |
| 9000 | 1.6695 | 1.8361 |
| 9900 | 1.6952 | 1.8328 |

- Val loss ban đầu (4.15) gần với mức đoán ngẫu nhiên trong bảng ký tự (ln 65 ≈ 4.17), sau đó giảm nhanh trong vài trăm vòng đầu rồi giảm chậm dần.
- Train loss in ra là loss của đúng một batch nên dao động; val loss là trung bình của 20 batch nên ổn định hơn và đáng tin hơn để đánh giá.
- Val loss cao hơn train loss khoảng 0.15 đến 0.17 và không tăng lên ở cuối huấn luyện, nên chưa có dấu hiệu học thuộc dữ liệu (overfitting) nghiêm trọng.

**Văn bản sinh ra** (prompt `Pacman`, trích đầu mỗi đoạn)

| Vòng lặp | Đoạn trích |
|---|---|
| 0 | `PacmanksmNk!t3HPKtvEmx iOjmz$iYB:ir-qRpvo&neRWpyxOFmKWqRM` |
| 1000 | `Pacmant, man bastel, / Thi, givatt, al I'taie, I neafer. Sto cooven thy you whith.` |
| 5000 | `Pacmany! / To Clares Romest the banishful callus you blood! / QUEEN MONELIO: / Pluch, if as your evoy tadsuch in winteles?` |
| 9000 | `Pacmantation deserve needl's stare we bruke noble go, when I am tamble procle? / O in the heath can fid Richmond change and wais you` |

**Nhận xét:** ở vòng 0 chỉ có ký tự ngẫu nhiên. Sau 1000 vòng đã có từ tiếng Anh và dấu câu. Từ khoảng vòng 2000 trở đi xuất hiện định dạng kịch bản (tên nhân vật viết hoa kèm dấu hai chấm). Đến cuối, văn bản có nhiều từ thật và tên nhân vật có thật như `GLOUCESTER`, `YORK`, `MENENIUS` nhưng nhiều từ vẫn bị bịa và câu chưa mạch lạc. Nguyên nhân một phần là ngữ cảnh chỉ có 10 ký tự và mạng nhỏ huấn luyện trên CPU.

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



