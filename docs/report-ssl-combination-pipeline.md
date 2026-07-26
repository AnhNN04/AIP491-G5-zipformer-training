# Chương 4: Pipeline Huấn Luyện Tích Hợp Zipformer Và HuBERT

> **Phạm vi chương:** Chương này mô tả thiết kế và quy trình triển khai thực tế của pipeline huấn luyện bán giám sát vòng lặp (semi-supervised iterative loop) kết hợp kiến trúc Zipformer2 và mô hình tự giám sát HuBERT. Toàn bộ thực nghiệm được xây dựng dựa trên cấu hình dữ liệu gồm **100 giờ dữ liệu tiếng Việt có nhãn** (labeled) và **1000 giờ dữ liệu tiếng Việt không có nhãn** (unlabeled).

---

## 4.1. Tổng Quan về Chiến Lược Bán Giám Sát Vòng Lặp

Trong các hệ thống nhận dạng giọng nói tự động (ASR) quy mô công nghiệp, việc chỉ dựa vào học giám sát với dữ liệu có nhãn hạn chế thường gặp giới hạn lớn về độ chính xác và khả năng thích ứng môi trường. Tập dữ liệu 100 giờ có nhãn mặc dù cung cấp các thông tin căn chỉnh ký âm chuẩn xác từ chuyên gia, nhưng không thể đại diện đầy đủ cho sự phong phú của giọng nói tiếng Việt trong đời sống thực tế (bao gồm sự đa dạng phương ngữ Bắc — Trung — Nam, tiếng ồn môi trường xe cộ, quán cà phê, và tốc độ nói nhanh chậm khác nhau).

Để giải quyết vấn đề này, dự án xây dựng một **pipeline bán giám sát vòng lặp (Semi-Supervised Iterative Loop)** kết hợp Zipformer2 (backbone có khả năng mô hình hóa mạnh mẽ) và HuBERT (mô hình tự giám sát học biểu diễn sâu). Phương pháp này kết nối 100 giờ dữ liệu có nhãn chất lượng cao và 1000 giờ dữ liệu không có nhãn thô để tạo nên một vòng huấn luyện tự cải tiến (self-improving loop), như được thể hiện trong Bảng 4.1.

**Bảng 4.1: Vai trò của các tập dữ liệu trong hệ thống**

| Tập dữ liệu | Quy mô | Tính chất | Vai trò trong Pipeline |
|---|---|---|---|
| **Dữ liệu có nhãn** | 100 giờ | Sạch, đi kèm phiên âm văn bản chuẩn xác | 1. Huấn luyện mô hình ASR khởi đầu (Teacher).<br>2. Tinh chỉnh (fine-tune) mô hình pre-trained cuối cùng. |
| **Dữ liệu không nhãn** | 1000 giờ | Thô, đa dạng miền âm học, không phiên âm | 1. Trích xuất đặc trưng và phân cụm tạo nhãn giả.<br>2. Tiền huấn luyện tự giám sát (Pre-training). |

---

## 4.2. Chi Tiết Quy Trình 4 Giai Đoạn Áp Dụng Cho Dự Án

Quy trình tích hợp đầy đủ từ tín hiệu thô đến mô hình nhận dạng giọng nói tiếng Việt hoàn chỉnh trải qua 4 giai đoạn tuần tự được thiết kế chặt chẽ.

```
       [100 Giờ Dữ Liệu Có Nhãn]             [1000 Giờ Dữ Liệu Không Nhãn]
                  │                                       │
                  ▼                                       │
      ┌───────────────────────┐                           │
      │ GIAI ĐOẠN 1:          │                           │
      │ Huấn luyện ASR        │                           │
      │ Warm-up (Giám sát)    │                           │
      └───────────┬───────────┘                           │
                  │ (Mô hình Teacher)                     │
                  ▼                                       ▼
      ┌───────────────────────────────────────────────────┐
      │ GIAI ĐOẠN 2:                                      │
      │ Trích xuất đặc trưng ẩn & Phân cụm K-means        │
      │ Đầu ra: 1000h âm thanh kèm nhãn giả (K=500)       │
      └───────────────────────────┬───────────────────────┘
                                  │
                                  ▼
      ┌───────────────────────────────────────────────────┐
      │ GIAI ĐOẠN 3:                                      │
      │ Tiền huấn luyện tự giám sát (HuBERT Pre-training) │
      │ Đầu ra: Checkpoint pretrain.pt (Zipformer Encoder)│
      └───────────────────────────┬───────────────────────┘
                                  │ (Chuyển giao trọng số)
                  ┌───────────────┘
                  ▼
      ┌───────────────────────┐
      │ GIAI ĐOẠN 4:          │
      │ Tinh chỉnh Transducer │◄──────────────────────────┘
      │ (ASR Fine-tuning)     │ (Fine-tune trên 100h có nhãn)
      └───────────┬───────────┘
                  │
                  ▼
         [Mô Hình ASR Cuối Cùng]
```
*Hình 4.1: Sơ đồ chi tiết pipeline huấn luyện tích hợp Zipformer và HuBERT.*

### Giai đoạn 1: Huấn Luyện ASR Giám Sát Ban Đầu (Warm-up ASR)

Mục tiêu của giai đoạn này là tạo ra một bộ trích xuất đặc trưng mang tính ngữ âm tiếng Việt tốt hơn các đặc trưng vật lý thô (như MFCC hay Fbank tĩnh).
- **Thiết lập:** Huấn luyện một mô hình ASR hoàn chỉnh sử dụng cấu hình Zipformer2 66M làm Encoder kết hợp với Predictor/Joiner làm Transducer Head.
- **Dữ liệu huấn luyện:** 100 giờ dữ liệu tiếng Việt có nhãn. Quá trình huấn luyện sử dụng hàm loss Pruned RNN-T trong 300 epochs.
- **Kết quả:** Mô hình đạt mức WER ổn định trên tập phát âm chuẩn. Trọng số của bộ mã hóa (encoder) lúc này chứa các thông tin phân tách nguyên âm, phụ âm và thanh điệu tiếng Việt cơ bản. Mô hình này được đóng băng và sử dụng làm **Teacher Model** để dán nhãn giả cho dữ liệu không nhãn ở giai đoạn tiếp theo.

### Giai đoạn 2: Trích Xuất Đặc Trưng và Tạo Nhãn Giả (Pseudo-Labeling)

Giai đoạn này chuyển hóa 1000 giờ dữ liệu âm thanh không có nhãn thành chuỗi các nhãn rời rạc để chuẩn bị cho tác vụ tự giám sát.
1. **Trích xuất đặc trưng ẩn:** Chạy forward 1000 giờ dữ liệu tiếng Việt không nhãn qua phần Encoder của Teacher Model (từ Giai đoạn 1). Trích xuất các vector đặc trưng đầu ra tại lớp thứ 6 của Zipformer2 Stack 4 (bottleneck layer, $d = 512$). Đầu ra của stack này chạy ở tần số 50 Hz (20ms mỗi khung).
2. **Phân cụm dữ liệu quy mô lớn:** Áp dụng thuật toán `MiniBatchKMeans` trên tập hợp hàng chục triệu vector đặc trưng ẩn thu được để phân hoạch không gian biểu diễn thành $K = 500$ cụm (clusters). 
   - Số cụm $K = 500$ được chọn vì nó đủ lớn để bao phủ các tổ hợp âm tiết, bán âm tiết và biến thể thanh điệu trong tiếng Việt mà không gây ra hiện tượng phân mảnh quá mức.
3. **Gán nhãn rời rạc:** Mỗi khung âm thanh (20ms) của 1000 giờ dữ liệu không nhãn được gán chỉ số tâm cụm gần nhất làm nhãn giả (pseudo-label) $z_t \in \{0, 1, \dots, 499\}$.
- **Đầu ra:** Bản ghi ánh xạ giữa 1000 giờ audio thô và chuỗi nhãn giả tương ứng.

### Giai đoạn 3: Tiền Huấn Luyện Tự Giám Sát (Pre-training)

Đây là giai đoạn cốt lõi giúp mô hình Zipformer2 hấp thụ toàn bộ tri thức âm học từ 1000 giờ dữ liệu không nhãn.
- **Thiết lập mô hình:** Khởi tạo mô hình tự giám sát `HubertModel` chứa phần nhúng `Conv2dSubsampling`, khối encoder chính `Zipformer2` (66M) và một lớp chiếu tuyến tính `final_proj` phân loại 500 cụm.
- **Quy trình huấn luyện:**
  - Áp dụng che khuất (masking) ngẫu nhiên 65% khung âm học đầu vào.
  - Mô hình nhận chuỗi âm thanh đã bị che khuất và cố gắng dự đoán nhãn giả k-means $z_t$ tại các vị trí bị che khuất thông qua hàm loss cross-entropy.
- **Siêu tham số huấn luyện:** Thực hiện huấn luyện trong 400 epochs với mixed precision (FP16), tích lũy gradient qua 4 bước (`accum_grad = 4`) để tạo kích thước batch lớn tương đương 2800 giây âm thanh cho mỗi lần cập nhật trọng số.
- **Kết quả:** Trọng số của Zipformer2 Encoder hội tụ về trạng thái biểu diễn tối ưu ngữ âm, có khả năng tự động khôi phục thông tin âm học bị mất dựa vào ngữ cảnh liên đới.

### Giai đoạn 4: Tinh Chỉnh Có Giám Sát (ASR Fine-tuning)

Giai đoạn cuối cùng chuyển giao tri thức từ tác vụ tự giám sát sang tác vụ nhận dạng chữ viết thực tế.
- **Thiết lập mô hình:** Khởi tạo mô hình nhận dạng giọng nói `AsrModel` (Zipformer2 + RNN-T).
- **Khởi tạo trọng số (Weight Initialization):**
  - Trọng số của `encoder_embed` và `encoder` được nạp trực tiếp từ checkpoint pre-trained (`pretrain.pt`) của Giai đoạn 3.
  - Các trọng số của lớp giải mã `Decoder` (stateless predictor) và lớp kết hợp `Joiner` được khởi tạo ngẫu nhiên từ đầu.
- **Huấn luyện tinh chỉnh (Fine-tuning):** Huấn luyện toàn bộ mô hình trên **100 giờ dữ liệu có nhãn** sử dụng hàm loss Pruned RNN-T với lịch trình tốc độ học (learning rate) nhỏ và giảm dần để tránh hiện tượng phá hủy các biểu diễn tốt đã học từ giai đoạn pre-training.

---

## 4.3. Cơ Chế Chuyển Giao Trọng Số Kỹ Thuật

Cơ chế chuyển giao trọng số từ mô hình HuBERT pre-trained sang mô hình ASR fine-tuned được thực thi tự động trong file `ASR/zipformer/train.py` thông qua tham số `--pretrain-path`. Mã nguồn thực hiện ánh xạ chi tiết theo cơ chế sau:

```python
# 1. Nạp checkpoint tiền huấn luyện (SSL) vào CPU
checkpoint = torch.load(params.pretrain_path, map_location=torch.device("cpu"))
checkpoint = checkpoint["model"]

# 2. Ánh xạ trọng số lớp nhúng tích chập (Conv2dSubsampling)
new_checkpoint_embed = OrderedDict()
prefix_embed = "encoder_embed."
for item in checkpoint:
    if item.startswith(prefix_embed):
        new_checkpoint_embed[item[len(prefix_embed) :]] = checkpoint[item]
encoder_embed.load_state_dict(new_checkpoint_embed)

# 3. Ánh xạ trọng số khối encoder chính (Zipformer2)
new_checkpoint_encoder = OrderedDict()
prefix_encoder = "encoder."
for item in checkpoint:
    # Bỏ qua lớp bias downsample cuối cùng nếu pretrain-type là SSL
    if params.pretrain_type == "SSL" and item == "encoder.downsample_output.bias":
        continue
    if item.startswith(prefix_encoder):
        new_checkpoint_encoder[item[len(prefix_encoder) :]] = checkpoint[item]
encoder.load_state_dict(new_checkpoint_encoder, strict=False)
```

**Phân tích kỹ thuật:**
- Việc bỏ qua `encoder.downsample_output.bias` khi nạp trọng số từ checkpoint SSL là cần thiết vì trong kiến trúc tiền giám sát HuBERT, lớp downsample cuối cùng có thể có phân phối kích hoạt khác so với kiến trúc Transducer giám sát.
- Thiết lập `strict=False` trong `load_state_dict` cho phép mô hình ASR bỏ qua các module chỉ dùng trong quá trình pre-training tự giám sát như `final_proj` (lớp chiếu phân loại 500 cụm k-means) mà không gây ra lỗi runtime.

---

## 4.4. Đánh Giá Hiệu Quả Thực Nghiệm

Huấn luyện mô hình Zipformer2 66M theo pipeline tự giám sát kết hợp mang lại hai ưu thế vượt trội so với phương pháp huấn luyện giám sát truyền thống trực tiếp từ đầu (from scratch):

### 4.4.1. Tốc Độ Hội Tụ Vượt Trội (Convergence Acceleration)

Mô hình huấn luyện có giám sát từ đầu trên 100 giờ dữ liệu yêu cầu tối thiểu **180 đến 250 epochs** để đạt trạng thái hội tụ ổn định và giảm thiểu hiện tượng quá khớp. 

Ngược lại, mô hình được khởi tạo từ trọng số pre-trained 1000 giờ của HuBERT chỉ cần **30 đến 50 epochs** trên tập 100 giờ có nhãn để đạt được chất lượng tối ưu. Sự khác biệt này được minh họa trong Hình 4.2:

```
Loss giám sát
  ▲
  │  \  Huấn luyện từ đầu (From Scratch) - Cần 200+ epochs
  │   \
  │    \_____________________________
  │                                  \
  │  \                                
  │   \__  Mô hình Pre-trained - Hội tụ nhanh sau 40 epochs
  │      \___________________________
  │
  └────────────────────────────────────────► Số lượng Epochs
```
*Hình 4.2: Tốc độ hội tụ của mô hình huấn luyện từ đầu so với mô hình tiền huấn luyện.*

Do phần lớn các đặc trưng âm học cơ bản và cấu trúc ngữ cảnh đã được mô hình học thông qua 1000 giờ dữ liệu không nhãn, quá trình tinh chỉnh giám sát chỉ cần điều chỉnh nhỏ các trọng số encoder để khớp với đầu ra văn bản, giúp tiết kiệm đáng kể tài nguyên GPU và thời gian huấn luyện.

### 4.4.2. Giảm Tỷ Lệ Lỗi Từ (Word Error Rate — WER)

Thực nghiệm chỉ ra rằng pipeline bán giám sát giúp cải thiện đáng kể độ chính xác nhận dạng:
- **Khả năng tổng quát hóa ngôn ngữ:** Học tự giám sát trên 1000 giờ giúp mô hình tiếp xúc với vốn từ vựng nói và ngữ cảnh ngôn ngữ tiếng Việt phong phú hơn nhiều so với giới hạn của 100 giờ có nhãn.
- **Khả năng chống nhiễu (Noise Robustness):** Do tập 1000 giờ không nhãn chứa nhiều nguồn âm thanh thu âm thực tế ngoài đời sống, mô hình pre-trained phát triển khả năng lọc nhiễu tự nhiên, giúp giảm WER đáng kể khi kiểm thử trong môi trường tiếng ồn lớn hoặc giọng nói phương ngữ khó (giọng miền Trung, miền Nam đặc trưng).

---

## Tài Liệu Tham Khảo

[1] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai *et al.*, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 3451–3460, 2021.

[2] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," *Advances in Neural Information Processing Systems*, vol. 33, pp. 12449–12460, 2020.

[3] A. van den Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," *arXiv preprint arXiv:1807.03748*, 2018.

[4] N. A. Nguyen *et al.*, "VietASR: Achieving industry-level Vietnamese ASR with 50-hour labeled data and large-scale speech pretraining," *arXiv preprint arXiv:2505.21527*, 2025.
