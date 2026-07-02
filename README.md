# 🩺 Federated Learning for Skin Lesion Classification

Dự án này triển khai và đánh giá các thuật toán **Học liên hợp (Federated Learning - FL)** trong lĩnh vực y tế, cụ thể là phân loại 7 nhóm tổn thương da trên bộ dữ liệu **HAM10000** và đánh giá kiểm thử độc lập (external test) trên bộ dữ liệu **ISIC 2019**. 

Dự án tập trung giải quyết các thách thức lớn của Học liên hợp trong thực tế lâm sàng:
1. **Sự không đồng nhất về dữ liệu (Data Heterogeneity / Non-IID)**: Sự khác biệt lớn về số lượng mẫu (Quantity Skew) và phân phối nhãn (Label Skew) giữa các bệnh viện/trạm y tế (clients).
2. **Sự mất cân bằng nhóm dữ liệu lâm sàng & Mức độ nghiêm trọng y khoa**: Ưu tiên tính chính xác đối với các loại bệnh nguy hiểm (như Melanoma - ung thư hắc tố da) so với các bệnh lành tính thông thường.
3. **Độ tin cậy của mô hình cục bộ**: Đánh giá tính không chắc chắn (uncertainty) của mô hình tại mỗi client thông qua phương pháp **Conformal Prediction**.

---

## 🚀 Tính năng nổi bật

- **Kiến trúc đồng bộ hóa phi tập trung (Decentralized Sync)**: Tận dụng **Hugging Face Hub** làm kênh truyền thông tin trung gian (Communication Hub) giữa Server và các Client (`hospital_AMD`, `hospital_NVIDIA`, `hospital_INTEL`). Các client và server trao đổi các tệp trọng số mô hình `.safetensors` và siêu dữ liệu định dạng `.json` bất đồng bộ/đồng bộ mà không cần kết nối trực tiếp.
- **Hỗ trợ nhiều phương pháp Học liên hợp**:
  - **FedAvg (Federated Averaging)**: Thuật toán FL cơ bản, trung bình cộng trọng số mô hình dựa trên số lượng mẫu dữ liệu của mỗi client.
  - **FedProx (Federated Proximal)**: Thêm thành phần điều chuẩn (proximal term) vào hàm mất mát tại client để hạn chế việc mô hình cục bộ bị lệch quá xa khỏi mô hình toàn cục (global model) do dữ liệu Non-IID.
  - **UCFedAvg (Uncertainty-Conformal Federated Averaging)**: Đánh giá độ tin cậy của client bằng cách đo lường độ phủ (coverage quality), kích thước tập dự đoán trung bình (average prediction set size) và độ không phù hợp (mean nonconformity) qua kỹ thuật Conformal Prediction trên tập validation cục bộ của từng client.
  - **CW-Risk-UCFedAvg (Class-Wise Risk-adjusted Uncertainty-Conformal Federated Averaging)**: Đề xuất cải tiến tiên tiến. Thực hiện trung bình hóa thông thường (UCFedAvg) đối với các tầng trích xuất đặc trưng (backbone) và **trung bình hóa theo từng lớp (class-wise)** đối với tầng phân loại (classifier head) dựa trên độ tin cậy của client đối với lớp đó kết hợp trọng số rủi ro y tế của từng bệnh (ví dụ: tăng trọng số cho các bệnh nguy cơ cao như Melanoma).
- **Quy trình kiểm thử Batch Evaluation chặt chẽ**: Batch evaluation tự động tải các mô hình tốt nhất từ Hugging Face qua các kịch bản (IID, Non-IID, Quantity Skew) và chạy thử nghiệm trên tập test độc lập ISIC 2019, vẽ biểu đồ so sánh ROC-AUC, ma trận nhầm lẫn chuẩn hóa (normalized confusion matrix), và tổng hợp kết quả ra file Excel/CSV.
- **Cam kết chống rò rỉ dữ liệu (No Data Leakage)**: Kiểm chứng nghiêm ngặt sự trùng lặp hình ảnh hoặc ca bệnh (lesion ID) giữa tập huấn luyện của client và tập test toàn cục, đảm bảo kết quả đánh giá khách quan và chính xác về mặt lâm sàng.

---

## 📂 Cấu trúc mã nguồn

Mã nguồn được tổ chức thành các Notebook tương ứng với từng giai đoạn và phương pháp thí nghiệm:

1. 💻 **[server-client-fedavg.ipynb](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/server-client-fedavg.ipynb)**: Triển khai kịch bản FL cơ bản sử dụng phương pháp **FedAvg**. Tự động khởi tạo mô hình toàn cục trên HF, điều phối quá trình huấn luyện và tổng hợp của các client.
2. 💻 **[server-client-fedfrox.ipynb](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/server-client-fedfrox.ipynb)**: Triển khai phương pháp **FedProx**. Client tối ưu hóa hàm loss bổ sung thêm proximal term với tham số $\mu = 0.01$, giúp ổn định hội tụ khi phân phối dữ liệu phân tán bị lệch lớn.
3. 💻 **[server-client-wc-risk-ucfedavg.ipynb](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/server-client-wc-risk-ucfedavg.ipynb)**: Mã nguồn của thuật toán cốt lõi **UCFedAvg** và **CW-Risk-UCFedAvg**. Tích hợp bộ tính toán Conformal Prediction, trích xuất metrics của từng client và phân phối trọng số tổng hợp theo lớp (class-wise aggregation).
4. 💻 **[evaluation.ipynb](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/evaluation.ipynb)**: Bộ script chạy Batch Evaluation trên Kaggle/Local. Tự động tải xuống các mô hình toàn cục ở các vòng tối ưu từ Hugging Face, đánh giá toàn diện và tạo các báo cáo chi tiết.
5. 📊 **[Data/](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/Data)**: Thư mục chứa cấu trúc chia dữ liệu huấn luyện và kiểm thử:
   - [Data_Source.md](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/Data/Data_Source.md): Đường dẫn tải dữ liệu gốc từ Kaggle (HAM10000 và ISIC 2019).
   - [split_validation_report.json](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/Data/split_validation_report.json): Báo cáo xác thực không trùng lặp (leakage) giữa train/val và test.
   - [global_test_distribution.json](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/Data/global_test_distribution.json): Phân phối nhãn của bộ test độc lập.
   - `iid/`, `noniid_label_skew/`, `quantity_skew/`: Các tệp CSV phân chia mẫu cho từng client trong mỗi kịch bản thí nghiệm.

---

## 🛠️ Thiết lập & Cấu hình thí nghiệm

### 1. Phân phối dữ liệu (Data Scenarios)
Thí nghiệm được mô phỏng trên 3 Client tương ứng với 3 bệnh viện giả định sở hữu các cấu hình phần cứng khác nhau: `hospital_AMD`, `hospital_NVIDIA`, `hospital_INTEL`.
- **IID**: Mẫu được xáo trộn và chia đều. Phân phối nhãn tại mỗi client tương tự nhau.
- **Quantity Skew**: Số lượng mẫu phân tán mất cân bằng trầm trọng (Nvidia nhận lượng mẫu lớn nhất, AMD vừa phải, Intel ít nhất).
- **Non-IID Label Skew**: Phân phối nhãn bị lệch mạnh. Một số client không có hoặc có rất ít mẫu của một vài lớp bệnh nhất định.

### 2. Các lớp bệnh tổn thương da (7 Classes)
Mô hình MobileNetV2 được tinh chỉnh để phân loại 7 nhóm bệnh:
- `nv`: Melanocytic nevi (Nốt ruồi hắc tố - Trọng số rủi ro: 1.00)
- `bkl`: Benign keratosis-like lesions (Dày sừng lành tính - Trọng số rủi ro: 1.10)
- `vasc`: Vascular lesions (Tổn thương mạch máu - Trọng số rủi ro: 1.20)
- `df`: Dermatofibroma (U sợi da - Trọng số rủi ro: 1.20)
- `bcc`: Basal cell carcinoma (Ung thư tế bào đáy - Trọng số rủi ro: 1.40)
- `akiec`: Actinic keratoses (Dày sừng ánh sáng - Trọng số rủi ro: 1.50)
- `mel`: Melanoma (Ung thư hắc tố - Trọng số rủi ro nguy hiểm nhất: 1.60)

---

## 📈 Cơ chế Thuật toán CW-Risk-UCFedAvg

Thuật toán hoạt động bằng cách phân tách cấu trúc mạng Neural:
1. **Backbone (Feature Extractor)**: 
   Sử dụng trọng số tổng hợp mức Client $w_k$:
   $$w_k \propto \left(\frac{n_k}{N}\right)^\beta \times \text{Reliability}_k \times \text{SampleConfidence}_k \times \text{RareFactor}_k$$
   Trong đó, $\text{Reliability}_k$ được tính toán từ các số liệu Conformal Prediction trên tập validation độc lập của client đó (phạt các client có tập dự đoán quá lớn hoặc độ phủ thực tế lệch xa độ phủ mục tiêu $1-\alpha$).

2. **Classifier Head (Lớp tuyến tính cuối)**:
   Tổng hợp trọng số riêng biệt cho từng lớp bệnh $c$:
   $$\alpha_{k,c} \propto \text{Reliability}_{k,c} \times \text{ClassConfidence}_{k,c} \times \text{RiskWeight}_c$$
   Trong đó, $\text{Reliability}_{k,c}$ đo lường khả năng dự đoán đúng riêng cho lớp $c$ của client $k$ (kết hợp Recall, F1, độ bao phủ conformal theo lớp). Điều này giúp bảo vệ hiệu năng phân loại các bệnh hiếm/nguy hiểm ngay cả khi client bị label skew nghiêm trọng.

---

## 🏁 Hướng dẫn chạy Thí nghiệm

### 1. Chuẩn bị môi trường
Cài đặt các gói thư viện cần thiết:
```bash
pip install torch torchvision safetensors huggingface_hub scikit-learn matplotlib pandas numpy openpyxl tqdm
```

### 2. Cấu hình Hugging Face Token
Vì hệ thống đồng bộ hóa qua Hugging Face Hub, bạn cần tạo các Repository Model tương ứng trên tài khoản cá nhân (ví dụ: một repo cho Server và các repo tương ứng cho client).
Thêm Hugging Face Token có quyền ghi (`WRITE`) vào biến môi trường hoặc cấu hình Kaggle Secrets:
- Tên Secret: `HF_TOKEN`

### 3. Quy trình thực hiện huấn luyện
1. Mở notebook của thuật toán muốn chạy (ví dụ: `server-client-wc-risk-ucfedavg.ipynb`).
2. Điều chỉnh các cấu hình ở đầu file:
   - `SERVER_REPO`: Repo của máy chủ toàn cục.
   - `CLIENT_CONFIGS`: Danh sách ID client và repo tương ứng của client.
   - `TRAINING_SCENARIO`: Kịch bản dữ liệu (`iid`, `noniid_label_skew` hoặc `quantityskew`).
3. Chạy các ô mã nguồn phần **Server** để khởi tạo mô hình vòng 0 bằng cách gọi hàm `init_global_model()`.
4. Chạy hàm điều phối tự động vòng huấn luyện Học liên hợp:
   - Hàm `run_auto_fl_training()` sẽ tự động chạy lặp qua các vòng huấn luyện:
     - Client tải mô hình toàn cục mới nhất về, huấn luyện cục bộ (1 epoch) trên tập dữ liệu của mình.
     - Client tính toán chênh lệch trọng số (delta) và metadata kết quả (bao gồm Conformal metrics) rồi đẩy lên HF.
     - Server chờ đủ các client hoàn thành, tải delta về, thực hiện tổng hợp (FedAvg / UCFedAvg / CW-Risk-UCFedAvg), đánh giá mô hình toàn cục trên tập test và cập nhật trạng thái vòng tiếp theo.
     - Cơ chế Early Stopping sẽ dừng huấn luyện nếu điểm số tối ưu hóa không cải thiện sau số vòng kiên nhẫn (`patience`).

### 4. Đánh giá kết quả kiểm thử (Evaluation)
Sau khi hoàn thành huấn luyện cho các thuật toán trên các kịch bản khác nhau, mở notebook **[evaluation.ipynb](file:///D:/Ph%C3%A2n%20t%C3%A1n/B%C3%A1o%20c%C3%A1o/evaluation.ipynb)**:
1. Điền đường dẫn các file trọng số `.safetensors` tốt nhất của các thí nghiệm vào danh sách `MODEL_PATHS`.
2. Chạy toàn bộ các ô để thực hiện batch evaluation trên bộ kiểm thử độc lập ISIC 2019.
3. Kết quả đánh giá sẽ được ghi nhận tại thư mục `/kaggle/working/batch_eval_results` (bao gồm ảnh so sánh, file dữ liệu Excel/CSV và file nén zip).
