# Chương 2: Kiến Trúc Mô Hình Nhận Dạng Giọng Nói — Zipformer

> **Phạm vi chương:** Chương này trình bày nền tảng lý thuyết và thiết kế kỹ thuật của mô hình acoustic encoder Zipformer2 cùng hệ thống giải mã RNN-Transducer được sử dụng trong dự án VietASR. Toàn bộ phân tích dựa trên **cấu hình 66 triệu tham số (phiên bản gốc)** được triển khai trong giai đoạn đầu của nghiên cứu.

---

## 2.1. Đặt Vấn Đề và Tổng Quan Kiến Trúc ASR

### 2.1.1. Bài Toán Nhận Dạng Giọng Nói Tự Động

Nhận dạng giọng nói tự động (Automatic Speech Recognition — ASR) là bài toán ánh xạ một chuỗi tín hiệu âm thanh rời rạc $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ sang chuỗi ký hiệu văn bản $\mathbf{y} = (y_1, y_2, \ldots, y_U)$, trong đó $T$ và $U$ là độ dài chuỗi âm thanh và chuỗi văn bản tương ứng, và thông thường $T \gg U$. Mục tiêu là tìm chuỗi văn bản có xác suất hậu nghiệm cao nhất:

$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid \mathbf{x})$$

Tiếng Việt đặt ra những thách thức đặc thù so với các ngôn ngữ châu Âu. Hệ thống ngữ âm tiếng Việt bao gồm 6 thanh điệu (ngang, huyền, sắc, hỏi, ngã, nặng), khiến cho một âm tiết cơ bản có thể mang tối đa 6 nghĩa hoàn toàn khác nhau. Ngoài ra, sự biến đổi phương ngữ giữa ba miền Bắc — Trung — Nam tạo ra các đặc trưng âm vị học khác biệt đáng kể. Trong bối cảnh dữ liệu có nhãn hạn chế (50 giờ âm thanh), việc lựa chọn kiến trúc encoder có khả năng trích xuất đặc trưng hiệu quả trở thành yếu tố then chốt quyết định chất lượng hệ thống.

### 2.1.2. Tổng Quan Pipeline Xử Lý End-to-End

Hệ thống ASR trong nghiên cứu này thực hiện quá trình chuyển đổi từ tín hiệu âm thanh thô sang văn bản tiếng Việt thông qua bốn giai đoạn tuần tự:

```
Âm thanh WAV (16kHz)
        │
        ▼
[Trích xuất Fbank]          → Tensor (N, T, 80) tại 100Hz
        │
        ▼
[Conv2dSubsampling]         → Tensor (N, T', 192) tại 50Hz
        │
        ▼
[Zipformer2 Encoder]        → Tensor (N, T'', 512) tại 25Hz
        │                     (Biểu diễn âm học H_enc)
        ▼
[RNN-T Decoder + Joiner]    → Phân phối xác suất token
        │
        ▼
[Giải mã (Beam Search)]     → Chuỗi token BPE
        │
        ▼
Văn bản tiếng Việt
```

*Hình 2.1: Pipeline xử lý end-to-end từ tín hiệu âm thanh đến văn bản.*

---

## 2.2. Đặc Trưng Âm Thanh Đầu Vào

### 2.2.1. Trích Xuất Log-Mel Filterbank (Fbank)

Bước tiền xử lý đầu tiên là biến đổi tín hiệu âm thanh thô thành ma trận đặc trưng tần phổ-thời gian. Quá trình này thực hiện qua ba bước:

**Bước 1 — Short-Time Fourier Transform (STFT):** Tín hiệu âm thanh rời rạc $s[n]$ được phân tích trong từng cửa sổ thời gian ngắn sử dụng hàm cửa sổ Hamming $w[n]$:

$$\text{STFT}(m, k) = \sum_{n=0}^{N-1} s[n + m \cdot H] \cdot w[n] \cdot e^{-j2\pi kn/N}$$

trong đó $N = 400$ mẫu (25ms tại 16kHz) là kích thước cửa sổ phân tích, $H = 160$ mẫu (10ms) là bước dịch chuyển giữa các cửa sổ liên tiếp, và $m$ là chỉ số khung thời gian. Tốc độ khung đầu ra là $f_{\text{input}} = 100\text{ Hz}$, tức một đặc trưng được tạo ra cho mỗi 10ms âm thanh.

**Bước 2 — Mel Filterbank:** Phổ biên độ $|\text{STFT}(m, k)|^2$ được chiếu lên thang tần số Mel thông qua $B = 80$ bộ lọc tam giác phân bố đều trên thang Mel:

$$\text{Mel}(f) = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)$$

Mỗi bộ lọc $h_b(k)$ ($b = 1, \ldots, 80$) tích hợp năng lượng tần số trong một dải tần trên thang Mel, tạo ra vector đặc trưng 80 chiều phù hợp với tri giác thính giác của con người.

**Bước 3 — Logarithm:** Logarithm tự nhiên được áp dụng để nén dải động:

$$\text{Fbank}(m, b) = \log\!\left(\sum_{k} |\text{STFT}(m, k)|^2 \cdot h_b(k) + \epsilon\right)$$

**Đầu ra:** Tensor $(N, T, 80)$ trong đó $N$ là kích thước batch, $T$ là số khung thời gian, và 80 là số chiều đặc trưng Mel.

### 2.2.2. Lớp Nhúng Tích Chập — Conv2dSubsampling

Trước khi đưa vào Zipformer encoder, tensor Fbank trải qua lớp `Conv2dSubsampling` nhằm đồng thời giảm chiều thời gian và ánh xạ sang không gian embedding phù hợp với kiến trúc encoder.

Lớp này bao gồm ba khối tích chập 2D xếp chồng với kernel $3 \times 3$ và stride $2 \times 2$ ở hai chiều đầu, kết hợp với một tầng ConvNeXt để làm phong phú thêm đặc trưng không gian. Biến đổi chiều thời gian tuân theo công thức:

$$T' = \left\lfloor \frac{T - 7}{2} \right\rfloor$$

Điều này tương ứng với việc giảm tốc độ khung từ $100\text{ Hz}$ xuống $50\text{ Hz}$ (một frame đầu ra ứng với mỗi 20ms âm thanh). Số chiều embedding đầu ra được đặt bằng $d_0 = 192$, bằng với chiều embedding của Stack 1 trong Zipformer encoder.

**Đầu ra của Conv2dSubsampling:** Tensor $(N, T', 192)$ được dùng làm đầu vào cho Zipformer2 Encoder.

---

## 2.3. Kiến Trúc Zipformer2 Encoder — Cấu Trúc U-Net Đa Tốc Độ Khung

### 2.3.1. Tư Tưởng Thiết Kế: Vượt Qua Giới Hạn $O(T^2)$ của Conformer

Conformer [1] là kiến trúc encoder tiêu chuẩn trong ASR hiện đại, kết hợp cơ chế Self-Attention với mạng tích chập. Tuy nhiên, cơ chế Multi-Head Self-Attention (MHSA) trong Conformer xử lý toàn bộ chuỗi ở một tốc độ khung cố định, dẫn đến độ phức tạp tính toán bậc hai theo độ dài chuỗi:

$$\text{Complexity}_{\text{MHSA}} = O(T^2 \cdot d)$$

Với tín hiệu tiếng nói dài (ví dụ 10 giây → $T = 1000$ khung tại 100Hz), chi phí tính toán trở nên đáng kể.

Zipformer2 [2] giải quyết vấn đề này bằng chiến lược **multi-rate downsampling**: chuỗi âm thanh được xử lý ở nhiều tốc độ khung khác nhau thông qua cấu trúc hình chữ U (U-Net). Tại tầng bottleneck (Stack 4), tốc độ khung giảm xuống 8 lần so với tốc độ vào:

$$\text{Complexity}_{\text{Stack 4}} = O\!\left(\left(\frac{T}{8}\right)^2 \cdot d\right) = O\!\left(\frac{T^2}{64} \cdot d\right)$$

Điều này không chỉ giảm chi phí tính toán 64 lần ở tầng biểu diễn sâu nhất, mà còn cho phép mô hình học đặc trưng ngữ nghĩa cấp cao ở độ phân giải thời gian thấp — nơi các đơn vị ngôn ngữ như âm tiết và từ được biểu diễn tự nhiên hơn.

### 2.3.2. Cấu Trúc 6 Stack Encoder và Skip Connections

Zipformer2 encoder được tổ chức thành 6 stack xử lý tuần tự, tạo thành cấu trúc U-Net với pha downsampling (Stack 1–4) và pha upsampling (Stack 4–6).

**Bảng 2.1: Cấu hình 6 Stack Encoder — Phiên bản 66M tham số gốc**

| Stack | Số Blocks | Tốc độ Khung | Dim $d$ | FF-dim | Số đầu Attention | CNN Kernel | Tham số xấp xỉ |
|:-----:|:---------:|:------------:|:-------:|:------:|:----------------:|:----------:|:--------------:|
| 1     | 2         | 50 Hz        | 192     | 512    | 4                | 31         | ~2.0M          |
| 2     | 2         | 25 Hz        | 256     | 768    | 4                | 31         | ~3.8M          |
| 3     | 3         | 12.5 Hz      | 384     | 1024   | 4                | 15         | ~8.5M          |
| 4     | 4         | 6.25 Hz      | 512     | 1536   | 8                | 15         | ~19.0M         |
| 5     | 3         | 12.5 Hz      | 384     | 1024   | 4                | 15         | ~8.5M          |
| 6     | 2         | 25 Hz        | 256     | 768    | 4                | 31         | ~3.8M          |

Việc chuyển đổi chiều embedding giữa các stack được thực hiện qua hàm `convert_num_channels`: nếu chiều tăng, các kênh mới được khởi tạo bằng giá trị không (zero-padding); nếu chiều giảm, các kênh cuối bị cắt bỏ (truncation). Kỹ thuật này loại bỏ nhu cầu sử dụng các lớp projection tuyến tính tốn kém giữa các stack.

**Cơ chế Skip Connection:** Hai kết nối tắt được thiết lập giữa các stack đối xứng trong cấu trúc U-Net:
- Stack 3 ($12.5\text{ Hz}$, $d=384$) → Stack 5 ($12.5\text{ Hz}$, $d=384$)
- Stack 2 ($25\text{ Hz}$, $d=256$) → Stack 6 ($25\text{ Hz}$, $d=256$)

Các skip connection này cộng biểu diễn từ pha downsampling vào pha upsampling tương ứng, đảm bảo thông tin chi tiết cấp thấp (âm vị, cấu trúc âm vị học) không bị mất trong quá trình nén thời gian.

**Đầu ra encoder:** Sau khi qua 6 stack, một lớp `SimpleDownsample` cuối cùng với `output_downsampling_factor=2` giảm tốc độ từ 50 Hz xuống 25 Hz. Đầu ra cuối cùng có chiều $d_{\max} = \max(192, 256, 384, 512, 384, 256) = 512$ và tốc độ khung 25 Hz (một frame = 40ms).

### 2.3.3. Cơ Chế Attention Đặc Biệt — Phân Tách MHSA Thành MHAW và SA

Trong MHSA chuẩn, ma trận attention weight được tính toán riêng cho mỗi module attention trong mỗi lớp. Zipformer giới thiệu một cải tiến quan trọng: **Multi-Head Attention Weights (MHAW)** tính ma trận attention một lần và chia sẻ cho tất cả các **Self-Attention (SA)** trong cùng một block.

Cho chuỗi đầu vào $\mathbf{X} \in \mathbb{R}^{T \times d}$, MHAW tính toán:

$$\mathbf{Q} = \mathbf{X} W_Q, \quad \mathbf{K} = \mathbf{X} W_K + \mathbf{R} W_R$$

$$A_{h,t,t'} = \text{softmax}\!\left(\frac{q_{h,t}^T k_{h,t'}}{\sqrt{d_q}}\right), \quad h = 1, \ldots, H$$

trong đó $\mathbf{R}$ là ma trận Compact Relative Positional Encoding, $d_q = 32$ là chiều của query/key mỗi head. Ma trận $A$ sau đó được tái sử dụng bởi tất cả các module SA trong block mà không cần tính toán lại, tiết kiệm đáng kể chi phí tính toán.

**Các tham số attention trong cấu hình 66M:**
- Query/Key dimension per head: $d_q = 32$ (hằng số cho tất cả stack)
- Value dimension per head: $d_v = 12$ (nhỏ hơn $d_q$, thiết kế có chủ đích)
- Positional encoding dimension per head: $d_{\text{pos}} = 4$
- Số đầu attention: $H \in \{4, 4, 4, 8, 4, 4\}$ theo từng stack

Stack 4 sử dụng $H=8$ thay vì $H=4$ như các stack khác. Tại tốc độ khung thấp nhất (6.25 Hz), mỗi frame đại diện cho 160ms âm thanh — đủ dài để chứa một âm tiết hoàn chỉnh. Với số đầu attention gấp đôi, mô hình có khả năng nắm bắt các phụ thuộc dài hạn phong phú hơn ở mức độ ngữ nghĩa này.

### 2.3.4. Cấu Trúc Nội Tại Một Zipformer Block

Mỗi Zipformer block trong một stack thực hiện biến đổi chuỗi $\mathbf{X} \rightarrow \mathbf{X}'$ thông qua một chuỗi các module được sắp xếp theo thứ tự cụ thể, khác biệt đáng kể so với Conformer truyền thống:

```
Đầu vào X
    │
    ├──→ [MHAW]                  → Ma trận attention A (dùng chung)
    ├──→ [FF Module 1]           hidden = (3/4) × d_ff
    ├──→ [NonLinear Attention]   dùng lại A từ MHAW
    ├──→ [Self-Attention 1]      dùng lại A, d_v = 12/head
    ├──→ [Conv1D 1]              depthwise, kernel k
    ├──→ [FF Module 2]           hidden = d_ff  (Middle FF)
    ├──→ [Self-Attention 2]      dùng lại A
    ├──→ [Conv1D 2]              depthwise, kernel k
    ├──→ [FF Module 3]           hidden = (5/4) × d_ff
    ├──→ [Bypass Module]         α ∈ (0,1) học được per-channel
    └──→ [BiasNorm]
         │
         ▼
      Đầu ra X'
```

**Thiết kế ba cấp Feed-Forward:** Sự biến đổi kích thước hidden dimension qua ba module FF không đồng đều là một thiết kế có chủ đích:

$$d_{\text{FF1}} = \frac{3}{4} d_{ff}, \quad d_{\text{FF2}} = d_{ff}, \quad d_{\text{FF3}} = \frac{5}{4} d_{ff}$$

*Ví dụ tại Stack 4 ($d=512$, $d_{ff}=1536$):*
- FF1: hidden = $1152$ — thu nhỏ, tiết kiệm tham số ở bước đầu
- FF2: hidden = $1536$ — biểu đạt đầy đủ
- FF3: hidden = $1920$ — mở rộng, tăng khả năng biểu đạt ở bước cuối

Tổng chi phí FF chiếm khoảng 62% chi phí mỗi block.

**Trường thụ cảm của Conv1D:** Kernel kích thước $k$ tại tốc độ khung $f$ tạo ra trường thụ cảm $k/f$ giây:
- Stack 1 ($50\text{ Hz}$, $k=31$): trường thụ cảm $= 620\text{ ms}$
- Stack 4 ($6.25\text{ Hz}$, $k=15$): trường thụ cảm $= 2.4\text{ giây}$

### 2.3.5. Các Thành Phần Chuẩn Hóa và Kỹ Thuật Đặc Biệt

**BiasNorm thay LayerNorm:** LayerNorm chuẩn hóa kích hoạt về phân phối đơn vị, vô tình xóa bỏ thông tin về độ lớn tuyệt đối của vector — thông tin quan trọng trong tín hiệu giọng nói (ví dụ: năng lượng âm thanh). BiasNorm thay thế bằng cách chỉ chia tỷ lệ mà không dịch chuyển:

$$\text{BiasNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \cdot e^{b}$$

trong đó $b$ là tham số vô hướng học được, cho phép mô hình điều chỉnh thang đo mà vẫn giữ lại thông tin hướng tương đối của vector.

**Bypass Module thay Residual Connection:** Kết nối tắt chuẩn $\mathbf{x} \leftarrow \mathbf{x} + f(\mathbf{x})$ được thay bằng Bypass module với trọng số vô hướng học được theo từng kênh $\boldsymbol{\alpha} \in (0, 1)^d$:

$$\mathbf{x}' = \boldsymbol{\alpha} \odot f(\mathbf{x}) + (1 - \boldsymbol{\alpha}) \odot \mathbf{x}$$

Điều này cho phép mô hình tự điều chỉnh mức độ "bypass" của thông tin gốc so với thông tin sau biến đổi, thay vì cộng cố định 1:1. Trong giai đoạn đầu huấn luyện, $\boldsymbol{\alpha} \approx \mathbf{0}$ (hành xử như identity), dần dần tăng lên khi mô hình hội tụ.

**Hàm kích hoạt SwooshR và SwooshL:** Zipformer thay thế Swish/GELU bằng hai biến thể tùy chỉnh:

$$\text{SwooshR}(x) = \log(1 + e^{x-4}) - 0.08x - 0.035$$
$$\text{SwooshL}(x) = \log(1 + e^{x-1}) - 0.08x$$

Các hàm này có vùng bão hòa âm rõ ràng hơn và gradient ổn định hơn trong quá trình huấn luyện với `ScaledAdam`.

**ScaledLinear:** Các lớp linear trong Zipformer được khởi tạo với tham số `initial_scale` nhỏ (thường $0.1$–$0.25$), nhằm đảm bảo output của mỗi lớp ở giai đoạn đầu có độ lớn nhỏ và ổn định, tránh hiện tượng gradient bùng nổ.

---

## 2.4. Bộ Tokenizer — SentencePiece BPE

### 2.4.1. Byte-Pair Encoding (BPE) cho Tiếng Việt

Đơn vị mô hình hóa của hệ thống là **subword token** được sinh ra từ thuật toán SentencePiece Unigram [3]. Mô hình tokenizer được huấn luyện trực tiếp từ văn bản phiên âm tiếng Việt trong tập dữ liệu huấn luyện có nhãn.

**Cấu hình tokenizer:**
- Kích thước từ vựng: $|\mathcal{V}| = 2000$ subword token
- Mô hình: Unigram Language Model
- Character coverage: $1.0$ (bao phủ toàn bộ ký tự tiếng Việt có dấu)
- Special tokens: `<blk>` (blank token, id=0) và `<sos/eos>` (start/end of sentence, id=1)

### 2.4.2. Đặc Thù Tiếng Việt trong Tokenization

Tiếng Việt là ngôn ngữ đơn âm tiết (monosyllabic) với các thanh điệu tích hợp vào âm tiết. Mỗi từ cơ bản thường là một âm tiết độc lập và không thể chia nhỏ hơn mà không mất nghĩa. Ví dụ, âm tiết "ban" với các thanh điệu khác nhau tạo ra sáu từ có nghĩa khác nhau hoàn toàn: "ban" (ngang), "bàn" (huyền), "bán" (sắc), "bản" (hỏi), "bãn" (ngã), "bạn" (nặng).

Do đặc điểm này, tokenizer SentencePiece với $|\mathcal{V}|=2000$ có xu hướng giữ nguyên hầu hết các âm tiết tiếng Việt như các token đơn lẻ. Một số từ ghép phổ biến (hai âm tiết) có thể được nhóm thành token duy nhất nếu tần suất xuất hiện đủ cao trong corpus huấn luyện.

### 2.4.3. Ảnh Hưởng đến Kiến Trúc RNN-T

Kích thước từ vựng $|\mathcal{V}| = 2000$ ảnh hưởng trực tiếp đến hai thành phần của mô hình: Decoder Embedding ($2000 \times 512 = 1{,}024{,}000$ tham số) và Joiner Output Linear ($512 \times 2000 = 1{,}024{,}000$ tham số). Tỷ lệ $T/U$ trong tiếng Việt — với câu trung bình 10 từ và 2 giây âm thanh ($T \approx 200$ frame sau subsampling) — xấp xỉ từ 15 đến 20, một thông số quan trọng trong thiết kế Pruned RNN-T Loss.

---

## 2.5. Mô Hình Giải Mã — RNN-Transducer (RNN-T)

### 2.5.1. Lý Thuyết Tổng Quát RNN-T

RNN-Transducer (RNN-T) [4] là framework giải mã end-to-end được lựa chọn trong nghiên cứu này do ưu điểm vượt trội trong ASR thời gian thực: không yêu cầu giả thiết độc lập điều kiện như CTC, và hỗ trợ giải mã từng bước mà không cần toàn bộ chuỗi đầu vào.

RNN-T bao gồm ba thành phần:

**Encoder (Acoustic Model):** Ánh xạ chuỗi đặc trưng âm học $\mathbf{x}$ sang chuỗi biểu diễn ẩn:
$$\mathbf{H}^{\text{enc}} = \text{Encoder}(\mathbf{x}) \in \mathbb{R}^{T \times d_{\text{enc}}}$$

**Predictor / Decoder (Language Model):** Tại vị trí $u$, nhận $u-1$ token đã phát sinh và tạo ra biểu diễn ngôn ngữ:
$$\mathbf{h}^{\text{dec}}_u = \text{Predictor}(\mathbf{y}_{1:u-1}) \in \mathbb{R}^{d_{\text{dec}}}$$

**Joiner:** Kết hợp biểu diễn âm học tại frame $t$ và biểu diễn ngôn ngữ tại vị trí $u$ để tính phân phối xác suất của token tiếp theo:
$$P(z_{t,u} \mid \mathbf{x}, \mathbf{y}_{1:u-1}) = \text{Softmax}\!\left(\text{Joiner}\!\left(\mathbf{h}^{\text{enc}}_t, \mathbf{h}^{\text{dec}}_u\right)\right)$$

trong đó $z_{t,u} \in \mathcal{V} \cup \{\varnothing\}$, với $\varnothing$ là blank token. Khi $z_{t,u} = \varnothing$, thời gian dịch chuyển $t \to t+1$; khi $z_{t,u} \in \mathcal{V}$, một token văn bản được phát sinh.

Xác suất chuỗi $\mathbf{y}$ được tính bằng cách marginalize over tất cả alignment hợp lệ:
$$P(\mathbf{y} \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t,u} P(z_{t,u} \mid \mathbf{x}, \mathbf{y}_{1:u-1})$$

### 2.5.2. Stateless Predictor (Prediction Network)

Phần lớn các hệ thống RNN-T truyền thống sử dụng LSTM có trạng thái ẩn làm Predictor, gây ra hai vấn đề chính: không thể song song hóa batch trong quá trình suy diễn và tăng độ trễ (latency). Nghiên cứu này áp dụng **Stateless Predictor** [5] — một kiến trúc phi hồi quy bao gồm:

**Lớp Embedding:**
$$\mathbf{e}_u = \text{Embedding}(y_{u-1}) \in \mathbb{R}^{d_{\text{dec}}}$$

với `Embedding(2000, 512)` gồm $2000 \times 512 = 1{,}024{,}000$ tham số.

**Lớp Tích Chập 1D theo Ngữ Cảnh (`context_size=2`):**
$$\mathbf{h}^{\text{dec}}_u = \text{ReLU}\!\left(\text{Conv1d}\!\left([\mathbf{e}_{u-1}, \mathbf{e}_u], \text{kernel}=2\right)\right)$$

với `Conv1d(512, 512, kernel=2, groups=128, bias=False)` — tích chập theo nhóm (grouped convolution) với 128 nhóm, cho phép mô hình ngữ cảnh trigram (nhìn 2 token trước) với chỉ $\frac{512 \times 512 \times 2}{128} = 4{,}096$ tham số cho lớp tích chập.

**Tổng tham số Predictor:** $1{,}024{,}000 + 4{,}096 \approx \mathbf{1.03M}$

**Lý do chọn `context_size=2` (trigram):** Thực nghiệm cho thấy context_size=2 cải thiện đáng kể WER so với bigram (context_size=1) với chi phí tính toán không đáng kể, đặc biệt trong tiếng Việt nơi các cụm từ phổ biến thường xuất hiện dưới dạng bigram âm tiết.

**Balancer Module:** Hai `Balancer` module được chèn sau Embedding và sau Conv1d nhằm ngăn chặn hiện tượng drift của độ lớn embedding trong quá trình huấn luyện dài.

### 2.5.3. Joiner Network

Joiner kết hợp biểu diễn từ encoder và decoder, sau đó ánh xạ sang phân phối xác suất trên từ vựng. Kiến trúc cụ thể được mô tả qua hai bước:

**Bước 1 — Chiếu tuyến tính với ScaledLinear (initial\_scale=0.25):**
$$\tilde{\mathbf{h}}^{\text{enc}}_{t} = W_{\text{enc}} \mathbf{h}^{\text{enc}}_t + \mathbf{b}_{\text{enc}}$$
$$\tilde{\mathbf{h}}^{\text{dec}}_{u} = W_{\text{dec}} \mathbf{h}^{\text{dec}}_u + \mathbf{b}_{\text{dec}}$$

với $W_{\text{enc}} \in \mathbb{R}^{d_j \times d_{\text{enc}}}$, $W_{\text{dec}} \in \mathbb{R}^{d_j \times d_{\text{dec}}}$, $d_j = 512$.

**Bước 2 — Kết hợp cộng và phi tuyến:**
$$\text{logit}_{t,u} = W_{\text{out}} \cdot \tanh\!\left(\tilde{\mathbf{h}}^{\text{enc}}_t + \tilde{\mathbf{h}}^{\text{dec}}_u\right)$$

với $W_{\text{out}} \in \mathbb{R}^{|\mathcal{V}| \times d_j}$, $|\mathcal{V}| = 2000$.

**Bảng 2.2: Chi tiết tham số Joiner (cấu hình 66M)**

| Module | Kích thước | Tham số |
|--------|-----------|:-------:|
| `encoder_proj` (ScaledLinear) | $512 \to 512$ | 262,656 |
| `decoder_proj` (ScaledLinear) | $512 \to 512$ | 262,656 |
| `output_linear` (Linear)      | $512 \to 2000$ | 1,026,000 |
| **Tổng Joiner**               |               | **~1.55M** |

Tham số `initial_scale=0.25` trong `ScaledLinear` đảm bảo logit đầu ra của Joiner ở giai đoạn đầu huấn luyện có độ lớn nhỏ, tránh gradient bùng nổ khi kết hợp hai luồng biểu diễn từ encoder và decoder.

---

## 2.6. Hàm Mất Mát — Pruned RNN-T Loss

### 2.6.1. Giới Hạn Của Standard RNN-T Loss

Hàm mất mát RNN-T chuẩn yêu cầu tính xác suất tích lũy trên toàn bộ lưới alignment $T \times U$. Tổng bộ nhớ GPU yêu cầu:

$$\text{Memory} = O(N \times T \times U \times |\mathcal{V}|)$$

Với batch $N=32$, $T=500$, $U=50$, $|\mathcal{V}|=2000$: $32 \times 500 \times 50 \times 2000 = 1.6 \times 10^9$ phần tử — hoàn toàn không khả thi trên GPU thông thường.

### 2.6.2. Pruned RNN-T Loss — Thuật Toán Ba Giai Đoạn

Pruned RNN-T [6] được phát triển trong framework k2/icefall giải quyết vấn đề này thông qua ba giai đoạn:

**Giai đoạn 1 — Ước Lượng Thô (Simple Loss):**

Thay vì dùng Joiner đầy đủ, một joiner đơn giản được xây dựng từ hai projection:
- `simple_am_proj`: $\text{Linear}(512, 2000)$ — chiếu encoder output
- `simple_lm_proj`: $\text{Linear}(512, 2000)$ — chiếu decoder output

Joiner đơn giản tính:
$$s_{\text{simple}}(t, u) = \text{simple\_am\_proj}(\mathbf{h}^{\text{enc}}_t) + \text{simple\_lm\_proj}(\mathbf{h}^{\text{dec}}_u)$$

Chi phí tính toán giảm nhờ tính chất additive: không cần tính ma trận $(t, u, d_j)$ mà chỉ cần hai tổng $(t, |\mathcal{V}|)$ và $(u, |\mathcal{V}|)$.

**Giai đoạn 2 — Cắt Tỉa Lưới (Grid Pruning):**

Từ phân phối alignment ước lượng, thuật toán xác định vùng alignment có xác suất cao nhất và giữ lại một dải hẹp với `prune_range=5` quanh đường chéo tối ưu:

$$\mathcal{S}_{\text{pruned}} = \left\{(t, u) : \left|u - u^*_t\right| \leq \left\lfloor\frac{\text{prune\_range}}{2}\right\rfloor\right\}$$

Kích thước lưới giảm từ $T \times U$ xuống $T \times 5$.

**Giai đoạn 3 — Ước Lượng Chính Xác (Pruned Loss):**

Joiner đầy đủ được tính chính xác nhưng chỉ trong vùng đã cắt tỉa $\mathcal{S}_{\text{pruned}}$:

$$\text{logit}_{t,u} = W_{\text{out}} \cdot \tanh\!\left(W_{\text{enc}}\mathbf{h}^{\text{enc}}_t + W_{\text{dec}}\mathbf{h}^{\text{dec}}_u\right), \quad (t,u) \in \mathcal{S}_{\text{pruned}}$$

Bộ nhớ yêu cầu giảm từ $O(T \times U)$ xuống $O(T \times 5)$, cho phép tăng batch size lên nhiều lần mà không cần thêm VRAM.

### 2.6.3. Tổng Hàm Mất Mát Huấn Luyện

Hàm mất mát tổng hợp kết hợp cả hai thành phần:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pruned}} + \lambda_{\text{simple}} \cdot \mathcal{L}_{\text{simple}}, \quad \lambda_{\text{simple}} = 0.5$$

Thành phần $\mathcal{L}_{\text{simple}}$ đóng vai trò quan trọng trong giai đoạn đầu huấn luyện để ổn định ước lượng vùng pruning; vai trò của nó giảm dần khi mô hình hội tụ và các vùng pruning trở nên chính xác hơn.

---

## 2.7. Các Chiến Lược Giải Mã

### 2.7.1. Greedy Search

Greedy Search là chiến lược giải mã đơn giản nhất, thực hiện lựa chọn tham lam tại mỗi bước thời gian. Tại frame $t$ và vị trí token $u$, token có xác suất hậu nghiệm cao nhất được chọn:

$$\hat{z}_{t,u} = \arg\max_{z \in \mathcal{V} \cup \{\varnothing\}} P(z \mid \mathbf{h}^{\text{enc}}_t, \mathbf{h}^{\text{dec}}_u)$$

**Quy tắc chuyển trạng thái:**
- Nếu $\hat{z}_{t,u} = \varnothing$: dịch chuyển thời gian $t \leftarrow t+1$, giữ nguyên $u$
- Nếu $\hat{z}_{t,u} \in \mathcal{V}$: phát sinh token, cập nhật $u \leftarrow u+1$, giữ nguyên $t$

Tham số `max_sym_per_frame` (thường đặt bằng 1) giới hạn số token được phát sinh tại một frame để tránh vòng lặp vô hạn trong trường hợp mô hình liên tục phát sinh token không phải blank.

**Mã giả thuật toán:**

```
Khởi tạo: hyp = [<blk>, <blk>], t = 0, sym_per_frame = 0
While t < T:
    If sym_per_frame >= max_sym_per_frame:
        t = t + 1; sym_per_frame = 0; continue
    logits = Joiner(H_enc[t], Decoder(hyp[-context_size:]))
    y = argmax(logits)
    If y == blank:
        t = t + 1; sym_per_frame = 0
    Else:
        hyp.append(y); sym_per_frame += 1
        Cập nhật Decoder state với context mới
Return hyp[context_size:]
```

**Đánh giá:** Greedy Search có độ phức tạp $O(T + U)$ và phù hợp cho các ứng dụng cần tốc độ suy diễn cao nhất. Tuy nhiên, do không duy trì tập giả thuyết, phương pháp này dễ mắc lỗi với các từ có âm vị tương đồng trong tiếng Việt (ví dụ: "sắc" / "sắt", "học" / "hộc").

### 2.7.2. Modified Beam Search

Modified Beam Search duy trì đồng thời `beam` ($= 4$) giả thuyết giải mã song song, tìm kiếm trong không gian giải pháp rộng hơn so với Greedy Search. Mỗi giả thuyết là một cặp $(\mathbf{y}_{\text{partial}}, \log P_{\text{acc}})$ trong đó $\mathbf{y}_{\text{partial}}$ là chuỗi token đã giải mã và $\log P_{\text{acc}}$ là log-xác suất tích lũy.

**Cơ chế batch hóa với k2.RaggedTensor:** Điểm khác biệt quan trọng của Modified Beam Search trong codebase là việc tính Joiner cho tất cả giả thuyết đồng thời trong một forward pass duy nhất, thay vì tuần tự từng giả thuyết:

```
Tại mỗi frame t:
1. Thu thập tất cả giả thuyết: {(hyp_i, log_p_i)} từ N utterance trong batch
2. Batch hóa decoder input: decoder_input = stack([hyp[-context:]...])
3. D = Decoder(decoder_input)               — một lần cho toàn bộ batch giả thuyết
4. E = encoder_proj(H_enc[t])              — chiếu encoder frame hiện tại
5. logits = Joiner(E, D, project_input=False)
6. log_probs = log_softmax(logits/τ) + log_p_accumulated
7. Lấy top-K token theo log_prob (dùng k2.RaggedTensor.topk)
8. Mở rộng / cập nhật tập giả thuyết
```

**Quy tắc mở rộng giả thuyết:**
- Token $\varnothing$ (blank): giả thuyết được giữ nguyên, frame $t \leftarrow t+1$
- Token $y \in \mathcal{V}$: token được thêm vào $\mathbf{y}_{\text{partial}}$, cập nhật Decoder state

**Tích hợp Context Graph:** Modified Beam Search hỗ trợ `context_graph` — một đồ thị hữu hạn trạng thái (FSA) mã hóa các từ khóa ưu tiên (hotwords). Khi một giả thuyết khớp với một prefix của hotword, log-xác suất của nó được cộng thêm `context_score` dương, tăng khả năng nhận dạng chính xác các thuật ngữ chuyên ngành.

**Điều chỉnh Temperature $\tau$:** Tham số nhiệt độ điều chỉnh độ nhọn của phân phối xác suất:
$$\log P_\tau(z) = \frac{\log P(z)}{\tau}$$

Với $\tau < 1$: phân phối nhọn hơn, ưu tiên mạnh token có xác suất cao nhất; với $\tau > 1$: phân phối phẳng hơn, tăng đa dạng giả thuyết.

**Đánh giá:** Modified Beam Search thường giảm WER từ 5–15% tương đối so với Greedy Search, với chi phí tính toán tăng tuyến tính theo `beam`. Trong dự án, `beam=4` là lựa chọn cân bằng giữa chất lượng nhận dạng và tốc độ suy diễn.

---

## 2.8. Tổng Hợp Số Tham Số Mô Hình (Cấu Hình 66M Gốc)

**Bảng 2.3: Phân bổ tham số theo thành phần**

| Thành phần | Mô tả kỹ thuật | Số tham số |
|-----------|---------------|:----------:|
| Conv2dSubsampling | 3×Conv2D + ConvNeXt, $80 \to 192$ dim | ~0.61M |
| Zipformer2 Stack 1 | 2 blocks, $d=192$, $d_{ff}=512$, $H=4$ | ~2.0M |
| Zipformer2 Stack 2 | 2 blocks, $d=256$, $d_{ff}=768$, $H=4$ | ~3.8M |
| Zipformer2 Stack 3 | 3 blocks, $d=384$, $d_{ff}=1024$, $H=4$ | ~8.5M |
| Zipformer2 Stack 4 | 4 blocks, $d=512$, $d_{ff}=1536$, $H=8$ | ~19.0M |
| Zipformer2 Stack 5 | 3 blocks, $d=384$, $d_{ff}=1024$, $H=4$ | ~8.5M |
| Zipformer2 Stack 6 | 2 blocks, $d=256$, $d_{ff}=768$, $H=4$ | ~3.8M |
| Stateless Decoder | Embedding(2000, 512) + Conv1d(groups=128) | ~1.03M |
| Joiner | encoder\_proj + decoder\_proj + output\_linear | ~1.55M |
| Simple AM/LM Proj | 2×Linear(512, 2000) — chỉ dùng khi huấn luyện | ~2.05M |
| **Tổng** | | **~66M** |

> **Ghi chú:** `simple_am_proj` và `simple_lm_proj` chỉ được sử dụng trong quá trình tính Pruned RNN-T Loss và không tham gia vào quá trình suy diễn (inference). Trong bản thực thi nhẹ hơn, hai thành phần này có thể được loại bỏ sau khi huấn luyện xong.

**Bảng 2.4: So sánh các biến thể kích thước Zipformer**

| Biến thể | `encoder_dim` | `num_layers` | `ff_dim` | Tổng tham số |
|----------|:-------------:|:------------:|:--------:|:------------:|
| Zipformer-Tiny | 192,192,...,192 | 2,2,2,2,2,2 | 512,512,... | ~12M |
| Zipformer-Small | 192,256,256,256,256,256 | 2,2,2,2,2,2 | 512,768,... | ~30M |
| **Zipformer-Original (dự án này)** | **192,256,384,512,384,256** | **2,2,3,4,3,2** | **512,768,1024,1536,...** | **~66M** |
| Zipformer-Large | 192,256,512,512,512,256 | 2,4,4,8,4,4 | 768,1024,1536,... | ~80M+ |

---

## Tài Liệu Tham Khảo

[1] A. Gulati, J. Qin, C.-C. Chiu *et al.*, "Conformer: Convolution-augmented transformer for speech recognition," *Proc. Interspeech*, pp. 5036–5040, 2020.

[2] Z. Yao, L. Liang, W. Kang *et al.*, "Zipformer: A faster and better encoder for automatic speech recognition," *arXiv preprint arXiv:2310.11230*, 2023.

[3] T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing," *Proc. EMNLP*, 2018.

[4] A. Graves, "Sequence transduction with recurrent neural networks," *arXiv preprint arXiv:1211.3711*, 2012.

[5] S. Kim, K. Lu, S. Tripuraneni *et al.*, "Reducing streaming ASR model delay with self alignment," *Proc. Interspeech*, 2021.

[6] F. Kuang, L. Liang, Z. Yao *et al.*, "Pruned RNN-T for fast, memory-efficient ASR training," *Proc. Interspeech*, 2022.

[7] D. S. Park, W. Chan, Y. Zhang *et al.*, "SpecAugment: A simple data augmentation method for automatic speech recognition," *Proc. Interspeech*, 2019.

[8] N. A. Nguyen *et al.*, "VietASR: Achieving industry-level Vietnamese ASR with 50-hour labeled data and large-scale speech pretraining," *arXiv preprint arXiv:2505.21527*, 2025.
