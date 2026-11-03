# Q4: Language Identification (RNN)

## Cài đặt
1. Cài Python 3.11
2. Tạo môi trường ảo: `py -3.11 -m venv venv` rồi bật `venv\Scripts\activate`
3. Cài thư viện: `pip install numpy==1.24.3 matplotlib torch`

## Chạy chương trình
Trong thư mục dự án:

    python autograder.py -q q4 --no-graphics

Kết quả mong đợi: `*** PASS: check_lang_id`, Q4: 5/5, test accuracy khoảng 86%.

## File chính
- models.py: lớp LanguageIDModel (RNN)
- train.py: hàm train_languageid
- losses.py: hàm languageid_loss
