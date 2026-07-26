# Chương 3: Phương Pháp Học Tự Giám Sát Với HuBERT

> **Phạm vi chương:** Chương này phân tích chi tiết về phương pháp học tự giám sát (Self-Supervised Learning — SSL) áp dụng mô hình HuBERT (Hidden-Unit BERT) trong dự án VietASR. Nội dung tập trung vào cơ chế trích xuất nhãn giả (pseudo-labels) qua phân cụm k-means, kiến trúc tích hợp Zipformer2-HuBERT, các cơ chế masking đặc thù của tiếng nói và thiết lập hàm mất mát cross-entropy trên các vùng bị che khuất.

---

## 3.1. Động Lực: Tại Sao Cần Học Tự Giám Sát cho ASR?

### 3.1.1. Thách Thức Dữ Liệu Có Nhãn Hạn Chế
Trong nhận dạng giọng nói tự động (ASR), việc xây dựng một bộ dữ liệu có nhãn chất lượng cao đòi hỏi chi phí cực kỳ lớn. Quá trình phiên âm giọng nói yêu cầu các chuyên gia ngôn ngữ nghe và viết lại chính xác từng từ, căn chỉnh thời gian (alignment), đồng thời phải xử lý các biến thể phương ngữ, tiếng ồn môi trường và hiện tượng nói lắp. 

Đối với tiếng Việt — một ngôn ngữ có tài nguyên trung bình và thấp (medium/low-resource) trên bản đồ công nghệ giọng nói toàn cầu, việc thiếu các corpus có nhãn quy mô lớn (hàng nghìn giờ) là rào cản chính. Trong dự án này, dữ liệu có nhãn sạch ban đầu chỉ giới hạn ở mức **50 giờ**. Việc huấn luyện trực tiếp một mạng nơ-ron sâu với cấu trúc phức tạp như mô hình Zipformer 66 triệu tham số trên tập dữ liệu nhỏ này chắc chắn dẫn đến hiện tượng quá khớp (overfitting), khả năng tổng quát hóa kém đối với các môi trường âm thanh thực tế.

### 3.1.2. Hướng Tiếp Cận Khai Thác Dữ Liệu Không Có Nhãn
Để vượt qua giới hạn của học giám sát (Supervised Learning), học tự giám sát (Self-Supervised Learning — SSL) nổi lên như một giải pháp đột phá. Ý tưởng cốt lõi của SSL là khai thác nguồn tài nguyên âm thanh khổng lồ không có nhãn sẵn có trên Internet (podcast, tin tức, talkshow, youtube). Thay vì dựa vào nhãn do con người gán, mô hình tự xây dựng các tác vụ dự báo (pretext tasks) từ chính cấu trúc nội tại của tín hiệu đầu vào để học cách biểu diễn đặc trưng âm học và ngôn ngữ.

Khái niệm này tương tự như mô hình BERT trong xử lý ngôn ngữ tự nhiên (NLP) học biểu diễn từ bằng cách đoán các từ bị che khuất trong câu thô, hay GPT học đoán từ tiếp theo. Với tín hiệu giọng nói liên tục, SSL hướng tới việc học cách biểu diễn cấu trúc âm học vi mô (micro-acoustic) lẫn thông tin ngữ cảnh ngữ nghĩa vĩ mô (macro-semantic) mà không cần một văn bản phiên âm nào.

### 3.1.3. Bối Cảnh SSL Trong Dự Án VietASR
Hệ thống VietASR áp dụng chiến lược học tự giám sát theo vòng lặp 4 giai đoạn nhằm tận dụng hiệu quả hơn 1000 giờ dữ liệu tiếng Việt không nhãn:
1. Huấn luyện một mô hình ASR cơ bản với 50 giờ dữ liệu có nhãn.
2. Trích xuất đặc trưng ẩn từ dữ liệu không nhãn bằng mô hình cơ bản và áp dụng phân cụm offline để tạo nhãn giả (pseudo-labels).
3. Thực hiện tiền huấn luyện tự giám sát (HuBERT pre-training) trên toàn bộ tập dữ liệu không nhãn bằng cách dự đoán các nhãn giả này tại các khung bị che khuất.
4. Tinh chỉnh (fine-tune) mô hình pre-trained với 50 giờ dữ liệu có nhãn để thu được mô hình nhận dạng giọng nói cuối cùng có chất lượng vượt trội.

---

## 3.2. Các Phương Pháp SSL Tiền Thân

Trước khi kiến trúc HuBERT được đề xuất, hai hướng tiếp cận chính trong SSL cho tiếng nói bao gồm học dự báo tương phản (Contrastive Learning) và lượng tử hóa biểu diễn online.

### 3.2.1. CPC (Contrastive Predictive Coding, 2019)
CPC [3] là một trong những mô hình SSL tiên phong cho dữ liệu dạng chuỗi thời gian như tiếng nói. Mô hình sử dụng một mạng tích chập để nén tín hiệu âm thanh thành chuỗi vector biểu diễn ẩn $\mathbf{z}_t$, sau đó dùng mạng hồi quy (như GRU) để tổng hợp thông tin ngữ cảnh lịch sử $\mathbf{c}_t$. 

Nhiệm vụ của CPC là dự đoán biểu diễn tương lai $\mathbf{z}_{t+k}$ dựa trên ngữ cảnh hiện tại $\mathbf{c}_t$. Hàm mất mát InfoNCE được sử dụng để tối đa hóa sự tương đồng giữa đại diện tương lai thực tế (positive sample) và cực tiểu hóa với các đại diện từ các thời điểm hoặc mẫu âm thanh khác (negative samples):

$$\mathcal{L}_{\text{InfoNCE}} = -\sum_{k} \log \frac{\exp(\mathbf{z}_{t+k}^T W_k \mathbf{c}_t)}{\exp(\mathbf{z}_{t+k}^T W_k \mathbf{c}_t) + \sum_{\tilde{\mathbf{z}} \in \mathcal{Z}_{\text{neg}}} \exp(\tilde{\mathbf{z}}^T W_k \mathbf{c}_t)}$$

Hạn chế lớn nhất của CPC là cấu trúc suy diễn ngữ cảnh một chiều (chỉ nhìn về quá khứ) và việc phụ thuộc lớn vào chất lượng của tập mẫu tiêu cực (negative samples) để tránh sụp đổ biểu diễn (representation collapse).

### 3.2.2. wav2vec 2.0 (Facebook AI, 2020)
wav2vec 2.0 [2] giải quyết bài toán ngữ cảnh một chiều bằng cách sử dụng mạng Transformer hai chiều (bidirectional). Đồng thời, mô hình xây dựng một bộ lượng tử hóa online (Vector Quantization — VQ) sử dụng phân phối Gumbel-Softmax để ánh xạ các đặc trưng liên tục từ mạng tích chập thành các vector rời rạc lấy từ một từ điển mã khóa (codebook) học được.

Nhiệm vụ tiền huấn luyện của wav2vec 2.0 là xác định đúng vector lượng tử hóa tương ứng của khung âm thanh bị che khuất trong một tập hợp chứa mẫu đúng và các mẫu tiêu cực khác từ cùng một câu phát âm. Hàm mất mát chính là sự kết hợp của loss tương phản (contrastive loss) và loss đa dạng hóa codebook (diversity loss). 

Mặc dù rất hiệu quả, wav2vec 2.0 gặp thách thức về mặt tối ưu hóa: việc huấn luyện song song cả mạng Transformer lẫn bộ lượng tử hóa rời rạc Gumbel-Softmax rất nhạy cảm với siêu tham số, dễ dẫn đến hiện tượng từ điển mã khóa bị sụp đổ (chỉ một số ít từ mã được sử dụng tích cực, phần còn lại bị bỏ hoang).

### 3.2.3. So Sánh Các Phương Pháp SSL Cho Tiếng Nói

Bảng 3.1 tóm tắt sự khác biệt cốt lõi về mặt phương pháp luận giữa CPC, wav2vec 2.0 và HuBERT.

**Bảng 3.1: So sánh đặc trưng kỹ thuật của các phương pháp SSL**

| Đặc trưng | CPC (2019) | wav2vec 2.0 (2020) | HuBERT (2021) |
|---|---|---|---|
| **Cơ chế biểu diễn** | Liên tục | Lượng tử hóa liên tục | **Rời rạc (Hidden Units)** |
| **Mục tiêu tối ưu** | Tương phản (InfoNCE) | Tương phản + Đa dạng hóa | **Dự báo nhãn (Cross-Entropy)** |
| **Tạo mục tiêu (Target)** | Dự báo tương lai | Lượng tử hóa online | **Phân cụm offline (K-means)** |
| **Kiến trúc mô hình** | CNN + GRU (Một chiều) | CNN + Transformer (Hai chiều) | **CNN + Zipformer/Transformer** |
| **Độ ổn định huấn luyện** | Khá ổn định | Kém ổn định (nhạy codebook) | **Rất ổn định** |

---

## 3.3. HuBERT: Hidden-Unit BERT cho Giọng Nói

### 3.3.1. Ý Tưởng Gốc: "Hidden Units" Làm Từ Vựng Tiềm Ẩn
HuBERT [1] được thiết kế dựa trên tư tưởng đưa bài toán tự giám sát tiếng nói về gần nhất với bài toán Masked Language Modeling (MLM) của BERT trong NLP. Trở ngại lớn nhất là tiếng nói không có cấu trúc từ vựng rời rạc (lexicon). Tín hiệu âm thanh là một dải sóng liên tục, biến thiên theo thời gian và không có ranh giới từ rõ ràng.

Để giải quyết vấn đề này, HuBERT đưa ra khái niệm **"Hidden Units"** (Đơn vị ẩn). Bằng cách áp dụng thuật toán phân cụm không giám sát (K-means) lên các đặc trưng âm thanh, mô hình nhóm các frame tín hiệu có tính chất âm học tương đồng lại với nhau. Mỗi cụm (cluster) được coi như một "từ vựng ẩn" (pseudo-phoneme hoặc hidden unit) có ID xác định. Chuỗi tín hiệu âm thanh liên tục giờ đây được chuyển đổi thành chuỗi các chỉ số cụm rời rạc:

$$\mathbf{z} = (z_1, z_2, \ldots, z_T), \quad z_t \in \{1, 2, \ldots, K\}$$

Nhiệm vụ của mô hình lúc này là học cách dự đoán chỉ số cụm $z_t$ tại các vị trí khung thời gian bị che khuất dựa trên ngữ cảnh xung quanh. Nghiên cứu thực nghiệm chỉ ra rằng, ngay cả khi các nhãn phân cụm K-means ban đầu có độ nhiễu lớn (độ chính xác âm vị thấp), sự **nhất quán về mặt phân bố** của chúng vẫn đủ mạnh để hướng dẫn mạng nơ-ron học được các biểu diễn âm học cực kỳ sâu sắc và đặc trưng ngữ âm cấu âm.

### 3.3.2. Sơ Đồ Luồng Dữ Liệu Của HuBERT

Hình 3.1 minh họa luồng xử lý và huấn luyện tự giám sát của mô hình HuBERT tích hợp kiến trúc Zipformer2:

```
                  Âm thanh không nhãn (.wav)
                              │
                              ▼
                       [Trích xuất Fbank]
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼ (Offline, 100Hz)                          ▼ (Online)
 [Offline K-means]                          [Time & Channel Masking]
        │                                           │ (Che khuất ~65% khung)
        ▼                                           ▼
[Nhãn giả z_t]                              [Khung âm thanh bị mask]
(Tần số: 50Hz)                                      │
        │                                           ▼
        │                                  [Conv2dSubsampling]
        │                                           │ (Downsample x2)
        │                                           ▼
        │                                  [Zipformer2 Encoder]
        │                                           │ (Mã hóa ngữ cảnh)
        │                                           ▼
        │                                 [Final Projection]
        │                                           │ (Ánh xạ sang K cụm)
        │                                           ▼
        └─────────────────────────────────► [Cross-Entropy Loss]
                                            (Chỉ tính trên vùng bị mask)
```
*Hình 3.1: Luồng dữ liệu huấn luyện tự giám sát Zipformer-HuBERT.*

---

## 3.4. Quá Trình Tạo Nhãn Giả (Pseudo-Labels) — Offline K-Means

HuBERT chia tách hoàn toàn quá trình tạo nhãn mục tiêu (Target Generation) và quá trình học biểu diễn (Representation Learning) thành hai bước độc lập: Phân cụm Offline và Tối ưu hóa Online.

### 3.4.1. Toán Học Về Thuật Toán Phân Cụm K-Means
Cho tập hợp các vector đặc trưng trích xuất từ dữ liệu âm thanh $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_M\}$, mục tiêu của K-means là phân chia $M$ vector này thành $K$ cụm khác nhau sao cho tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm tương ứng (inertia) là nhỏ nhất:

$$\arg\min_{\mathbf{S}} \sum_{i=1}^{K} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

trong đó $S_i$ là tập các điểm thuộc cụm $i$, và $\boldsymbol{\mu}_i$ là tâm của cụm $i$. Quá trình tối ưu hóa được thực hiện thông qua hai bước lặp:
- **E-Step (Gán cụm):** Mỗi điểm dữ liệu được gán cho tâm cụm gần nhất:
  $$s_t = \arg\min_{i \in \{1,\ldots,K\}} \|\mathbf{x}_t - \boldsymbol{\mu}_i\|^2$$
- **M-Step (Cập nhật tâm cụm):** Cập nhật tọa độ tâm cụm bằng trung bình cộng các điểm trong cụm đó:
  $$\boldsymbol{\mu}_i = \frac{1}{|S_i|} \sum_{\mathbf{x} \in S_i} \mathbf{x}$$

### 3.4.2. Huấn Luyện Vòng Lặp 1 (Iteration 1) — K-Means Trên Đặc Trưng MFCC
Trong vòng lặp đầu tiên, khi hệ thống chưa có bất kỳ mô hình biểu diễn âm học nào, nhãn giả được sinh ra từ các đặc trưng vật lý truyền thống:
- **Đặc trưng trích xuất:** Hệ số MFCC (Mel-Frequency Cepstral Coefficients) 39 chiều bao gồm: 13 hệ số tĩnh, 13 hệ số delta (sai phân bậc một) và 13 hệ số delta-delta (sai phân bậc hai). Tốc độ trích xuất là 100 Hz.
- **Số lượng cụm:** Đặt cố định $K = 100$. Số cụm nhỏ giúp K-means dễ dàng hội tụ và gom nhóm các đặc trưng âm học thô (như nguyên âm kéo dài, khoảng lặng, phụ âm nổ).
- **Kết quả:** Chuỗi nhãn giả $\mathbf{z}^{(1)}$ có độ phân giải thời gian 10ms.

### 3.4.3. Huấn Luyện Vòng Lặp Tiếp Theo (Iteration 2+) — K-Means Trên Biểu Diễn Ẩn
Sau khi hoàn thành huấn luyện HuBERT vòng lặp 1, mô hình đã học được cách ánh xạ các đặc trưng thô thành các biểu diễn ngữ cảnh có tính chọn lọc cao.
- **Đặc trưng trích xuất:** Chạy forward dữ liệu qua mô hình HuBERT vòng 1, trích xuất biểu diễn ẩn tại một tầng trung gian của encoder (thường chọn lớp thứ 6 hoặc lớp thứ 12 tùy thuộc độ sâu của mô hình).
- **Số lượng cụm:** Tăng lên $K = 500$ (hoặc lên đến 1000). Do biểu diễn ẩn chứa thông tin ngữ âm tốt hơn, việc tăng số lượng cụm cho phép mô hình phân tách các âm vị tiếng Việt một cách chi tiết hơn (ví dụ phân biệt rõ thanh hỏi và thanh ngã, các phụ âm đầu tương tự nhau).
- **Thực thi tính toán:** Do kích thước dữ liệu quá lớn (hàng triệu vector đặc trưng ẩn 512 chiều), thuật toán K-means chuẩn không thể lưu trữ hết trên RAM. Repo VietASR giải quyết bằng cách áp dụng **MiniBatchKMeans** từ thư viện `scikit-learn` trong file `learn_kmeans.py`, thực hiện phân cụm theo các batch ngẫu nhiên cỡ nhỏ (ví dụ `batch_size = 10000`) nhằm giảm thiểu yêu cầu bộ nhớ mà vẫn đảm bảo chất lượng phân cụm.

### 3.4.4. Tại Sao Offline Phân Cụm Tốt Hơn Lượng Tử Hóa Online?
- **Tránh sụp đổ biểu diễn (Representation Collapse):** Trong wav2vec 2.0, nếu bộ lượng tử hóa online bị hội tụ lệch, mô hình sẽ rơi vào trạng thái chỉ dự đoán một vài mã khóa cố định. Với offline clustering, nhãn mục tiêu được cố định trước khi bước vào epoch huấn luyện, đảm bảo luôn có sự phân bố nhãn đa dạng cho mô hình học tập.
- **Độ ổn định huấn luyện:** Việc tách biệt hoàn toàn giúp tối ưu hóa đồ thị tính toán. Quá trình tính gradient chỉ tập trung vào việc học biểu diễn của encoder mà không phải cân đối giữa học biểu diễn và phân tách codebook.

---

## 3.5. Cơ Chế Masking

Cơ chế masking trong HuBERT được thiết kế để ép buộc mô hình phải dựa vào thông tin ngữ cảnh để tái tạo lại thông tin bị mất, từ đó phát triển cả khả năng nhận diện âm học lẫn khả năng mô hình hóa ngôn ngữ nội tại.

### 3.5.1. Time Masking (Che Khuất Theo Chiều Thời Gian)
- **Xác suất bắt đầu mask:** Mô hình chọn ngẫu nhiên một tỷ lệ `mask_prob = 0.65` các khung thời gian để làm điểm bắt đầu của một span bị che khuất.
- **Độ dài span che khuất:** Để tránh việc mô hình chỉ học cách nội suy tuyến tính các frame lân cận (do tín hiệu giọng nói giữa các khung 10ms có độ tương quan cực kỳ cao), HuBERT sử dụng span masking với `mask_length = 10` frames (tương đương 100ms âm thanh).
- **Cơ chế áp dụng:** Toàn bộ các khung thời gian thuộc span được chọn sẽ bị ghi đè bởi một vector tham số học được `mask_emb` có cùng kích thước với chiều embedding ($192$ chiều).
- **Phân phối lựa chọn:** Chế độ mặc định là `mask_selection = "static"`, tức độ dài span luôn cố định bằng 10. File `hubert_ce.py` cũng hỗ trợ các phân phối khác như "uniform", "normal", hoặc "poisson" để tăng độ đa dạng.

### 3.5.2. Channel Masking (Che Khuất Theo Chiều Đặc Trưng)
Để tăng độ khó cho tác vụ pre-training, ngoài việc che khuất theo trục thời gian, HuBERT có thể áp dụng che khuất theo trục đặc trưng (các kênh của vector embedding):
- **Tham số:** `mask_channel_prob` xác định xác suất một nhóm kênh bị che khuất, và `mask_channel_length` quyết định số kênh bị ghi đè bằng giá trị 0 tại khung thời gian đó.
- **Mục tiêu:** Tương tự như kỹ thuật Frequency Masking của SpecAugment [7], channel masking buộc mô hình không được phụ thuộc vào một dải tần số hoặc một nhóm đặc trưng cố định nào để dự đoán nhãn giả.

### 3.5.3. Sự Nhất Quán Của Masking Trong Batch
Để quá trình huấn luyện song song trên GPU đạt hiệu năng tối ưu, hàm `compute_mask_indices` hỗ trợ tham số `require_same_masks = True`. Kỹ thuật này đảm bảo mọi mẫu âm thanh trong một batch huấn luyện có chính xác số lượng khung bị che khuất bằng nhau bằng cách ngẫu nhiên thêm hoặc bớt các điểm mask. Điều này giúp loại bỏ sự lệch kích thước tensor trong tính toán cross-entropy.

---

## 3.6. Kiến Trúc Mô Hình HuBERT trong Dự Án

Kiến trúc HuBERT được triển khai trong dự án VietASR mang những nét đặc trưng riêng, tối ưu hóa cho cấu hình Zipformer2 66M tham số.

### 3.6.1. Thay Thế Transformer Bằng Zipformer2
Trong phiên bản HuBERT gốc của Facebook AI, phần encoder sử dụng mạng Transformer chuẩn (12 layers cho bản Base và 24 layers cho bản Large). Trong dự án này, để tối đa hóa hiệu năng và tốc độ xử lý, mạng Transformer được thay thế hoàn toàn bằng **Zipformer2 Encoder** với cấu hình 6 stack và 66 triệu tham số. 

Sự thay đổi này mang lại lợi thế kép: mô hình tự giám sát thừa hưởng khả năng xử lý đa tốc độ khung hình hiệu quả của Zipformer, giảm tải tính toán trên GPU trong khi vẫn học được các đặc trưng ngữ cảnh hai chiều chất lượng cao.

### 3.6.2. Các Thành Phần Chi Tiết Trong `HubertModel` (`hubert_ce.py`)
Mô hình tự giám sát được định nghĩa qua lớp `HubertModel` kế thừa từ `nn.Module`:
- **`encoder_embed` (Conv2dSubsampling):** Chuyển đổi Fbank 80 chiều đầu vào thành tensor embedding 192 chiều, đồng thời giảm tần suất từ 100Hz xuống 50Hz.
- **`mask_emb` (Parameter):** Vector tensor kích thước 192 chiều, được khởi tạo ngẫu nhiên theo phân phối đều, dùng để thay thế cho các vùng bị mask trong quá trình forward.
- **`encoder` (Zipformer2):** Nhận đầu vào 192 chiều, thực hiện xử lý qua 6 stack encoder và trả về biểu diễn ẩn cuối cùng 512 chiều ở tần số 25Hz.
- **`layer_norm` (LayerNorm):** Áp dụng chuẩn hóa lớp lên các đặc trưng embedding trước khi đưa vào encoder để ổn định phân phối giá trị.
- **`final_proj` (Linear):** Lớp chiếu tuyến tính ánh xạ từ biểu diễn ẩn 512 chiều của encoder lên số chiều tương ứng với số lượng cụm K-means mục tiêu (`num_classes` = 504, trong đó 500 là số cụm và 4 lớp đệm bổ sung).

### 3.6.3. Thiết Lập Tham Số Huấn Luyện Tự Giám Sát
Các tham số cấu hình chính trong file `pretrain.py`:
- `mask_before_cnn = True`: Áp dụng masking trực tiếp trên dữ liệu Fbank đầu vào trước khi qua lớp subsampling (giúp bảo vệ mô hình khỏi việc rò rỉ thông tin biên thông qua phép tích chập).
- `pred_masked_weight = 1.0`: Trọng số tối ưu hóa cho vùng bị che khuất.
- `pred_nomask_weight = 0.0`: Trọng số cho vùng không bị che khuất. Bằng việc đặt giá trị này bằng 0, mô hình hoàn toàn không tính loss trên các khung không bị mask, giải phóng mạng khỏi việc tối ưu hóa các dự báo tầm thường (trivial predictions).
- `logit_temp = 0.1`: Nhiệt độ chia tỷ lệ logits trước khi tính softmax.

---

## 3.7. Hàm Mất Mát — Masked Prediction với Cross-Entropy

Mặc dù có tên gọi liên quan đến "BERT", HuBERT sử dụng hàm mất mát phân loại Cross-Entropy chuẩn thay vì các loss tương phản phức tạp.

### 3.7.1. Công Thức Hàm Mất Mát Chính
Gọi $\mathcal{M} \subset \{1, \ldots, T''\}$ là tập hợp các chỉ số khung thời gian sau subsampling bị che khuất trong câu phát âm. Với mỗi khung $t \in \mathcal{M}$, mô hình sinh ra vector logit đầu ra $\mathbf{o}_t \in \mathbb{R}^K$ thông qua lớp chiếu `final_proj`. 

Xác suất dự đoán cụm $k$ tại thời điểm $t$ được tính bằng softmax có hiệu chỉnh nhiệt độ $\tau$:

$$P(k \mid \mathbf{h}_t) = \frac{\exp(o_{t,k} / \tau)}{\sum_{j=1}^{K} \exp(o_{t,j} / \tau)}$$

Hàm mất mát phân loại cross-entropy trên tập các vị trí bị masked được định nghĩa:

$$\mathcal{L}_{\text{masked}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P(z_t \mid \mathbf{h}_t)$$

trong đó $z_t \in \{1, \ldots, K\}$ là chỉ số cụm K-means mục tiêu của khung thời gian đó.

### 3.7.2. Tổng Hàm Mất Mát Tối Ưu Hóa
Hàm mất mát cuối cùng được tính toán trong hàm `compute_loss()` của file `hubert_ce.py` bao gồm hai thành phần:

$$\mathcal{L}_{\text{total}} = w_m \mathcal{L}_{\text{masked}} + \alpha \mathcal{L}_{\text{pen}}$$

- $w_m = 1.0$: Trọng số phần loss masked.
- $\mathcal{L}_{\text{pen}} = \frac{1}{B \cdot T \cdot d} \sum_{b,t,d} f_{b,t,d}^2$: Hình phạt chuẩn $L_2$ áp dụng lên vector đặc trưng sau subsampling (feature penalty).
- $\alpha = 10.0$ (tương ứng với `--loss-weights 10`): Trọng số của feature penalty, có vai trò kiểm soát không cho độ lớn của các vector kích hoạt tăng quá cao, giữ cho không gian embedding phân bố ổn định quanh gốc tọa độ.

### 3.7.3. Vai Trò Của Nhiệt Độ Logit $\tau = 0.1$
Việc đặt $\tau = 0.1$ (nhỏ hơn nhiều so với mặc định $\tau = 1.0$) có ý nghĩa quan trọng:
- Làm tăng độ dốc của hàm softmax, đẩy phân phối xác suất đầu ra của mô hình trở nên rất nhọn (sharp).
- Do nhãn giả K-means là nhãn cứng (one-hot target), nhiệt độ thấp buộc mô hình phải đưa ra các dự đoán mang tính quyết đoán cao, thúc đẩy quá trình học biểu diễn ngữ âm diễn ra nhanh và rõ ràng hơn.

---

## 3.8. Alignment Giữa Fbank và K-Means Labels

Một vấn đề kỹ thuật quan trọng trong thực thi HuBERT là sự lệch tần số giữa các luồng dữ liệu.

### 3.8.1. Sự Khác Biệt Tần Số Khung
- Đặc trưng đầu vào Fbank có tốc độ khung 100 Hz (1 frame mỗi 10ms).
- Sau khi đi qua lớp `Conv2dSubsampling` của encoder, tốc độ khung giảm đi 2 lần, còn 50 Hz (1 frame mỗi 20ms).
- Nhãn giả K-means được tạo ra ở tốc độ khung 50 Hz (được cấu hình bởi tham số `--label-rate 50`).

### 3.8.2. Công Thức Căn Chỉnh Trong `forward_targets()`
Để đảm bảo mỗi frame biểu diễn ẩn đầu ra của encoder khớp chính xác với nhãn giả tương ứng, hàm `forward_targets` thực hiện căn chỉnh dựa trên tỷ lệ `feat2tar_ratio`:

$$\text{feat2tar\_ratio} = \frac{\text{label\_rate} \times \text{feature\_ds\_rate}}{\text{sample\_rate}} = \frac{50 \times 2}{100} = 1.0$$

Với tỷ lệ bằng $1.0$, chỉ số khung thời gian của encoder và chỉ số nhãn giả K-means được ánh xạ trực tiếp 1:1.

```python
# Cắt bớt phần dư thừa ở cuối chuỗi đặc trưng nếu có sự lệch độ dài nhỏ
feat_tsz = features.size(2)
targ_tsz = min([t.size(1) for t in target_list])
if self.feat2tar_ratio * feat_tsz > targ_tsz:
    feat_tsz = int(targ_tsz / self.feat2tar_ratio)
    features = features[..., :feat_tsz]

# Trích xuất nhãn giả tương ứng
target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
target_list = [t[:, target_inds.long()] for t in target_list]
```

Cơ chế này loại bỏ hoàn toàn hiện tượng lệch pha (phase-shift) giữa âm thanh đầu vào và chuỗi nhãn giả mục tiêu trong quá trình pre-training.

---

## 3.9. Quy Trình Huấn Luyện SSL Đa Vòng Lặp

Quy trình pre-training tự giám sát với HuBERT được thực hiện qua nhiều vòng lặp để liên tục tinh chỉnh chất lượng nhãn giả và biểu diễn ẩn.

```
VÒNG LẶP 1:
   Đặc trưng âm học thô (MFCC) ──► Phân cụm K-means (K=100)
                                          │
                                          ▼
   Huấn luyện tự giám sát ───────► Mô hình HuBERT Vòng 1
                                          │
                                          ▼
VÒNG LẶP 2:
   Trích xuất đặc trưng ẩn từ lớp trung gian của HuBERT Vòng 1
                                          │
                                          ▼
   Phân cụm K-means quy mô lớn (K=500) ──► Nhãn giả chất lượng cao
                                          │
                                          ▼
   Huấn luyện tự giám sát ───────► Mô hình HuBERT Vòng 2 (Mạnh nhất)
```
*Hình 3.2: Quy trình huấn luyện HuBERT đa vòng lặp.*

### 3.9.1. Thiết Lập Siêu Tham Số Huấn Luyện Trong `pretrain.py`
Quá trình huấn luyện tự giám sát trên dữ liệu không nhãn quy mô lớn đòi hỏi tài nguyên tính toán mạnh mẽ và các thiết lập tối ưu:
- **Số lượng Epochs:** 400 epochs. SSL yêu cầu thời gian hội tụ dài hơn nhiều so với học giám sát do không gian nhãn giả rộng và không có sự định hướng trực tiếp từ văn bản thật.
- **Tối ưu hóa Gradient Accumulation:** `--accum-grad 4`. Để mô phỏng kích thước batch cực lớn cần thiết cho việc học tự giám sát ổn định mà không làm tràn bộ nhớ VRAM, các gradient được tích lũy qua 4 bước forward/backward trước khi cập nhật trọng số.
- **Thuật toán tối ưu (Optimizer):** Sử dụng `ScaledAdam` phối hợp với lịch trình học `Eden scheduler` tương tự như phần huấn luyện có giám sát. Điều này đảm bảo tốc độ hội tụ ổn định trên kiến trúc Zipformer2.
- **Tối ưu hóa bộ nhớ:** Sử dụng mixed precision training (`--use-fp16 1`) để giảm tải bộ nhớ GPU và tăng tốc độ tính toán gấp đôi.

---

## Tài Liệu Tham Khảo

[1] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 3451–3460, 2021.

[2] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," *Advances in Neural Information Processing Systems*, vol. 33, pp. 12449–12460, 2020.

[3] A. van den Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," *arXiv preprint arXiv:1807.03748*, 2018.

[4] N. A. Nguyen *et al.*, "VietASR: Achieving industry-level Vietnamese ASR with 50-hour labeled data and large-scale speech pretraining," *arXiv preprint arXiv:2505.21527*, 2025.
