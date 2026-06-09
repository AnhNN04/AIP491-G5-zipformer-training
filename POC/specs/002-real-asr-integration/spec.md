# Feature Specification: Real ASR Serving Integration

**Feature Branch**: `002-real-asr-integration`

**Created**: 2026-06-09

**Status**: Draft

**Input**: User description: "Bóc tách và tích hợp lõi mô hình thực tế từ thư mục ASR/ và SSL/ vào asr-server để thực hiện suy luận thật (không dùng mock) cho cả luồng HTTP REST và WebSocket Streaming."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Nhận dạng giọng nói theo lô (Batch ASR) (Priority: P1)

Người dùng tải lên một tệp âm thanh (ví dụ: MP3) từ giao diện Web, hệ thống tự động lưu trữ và thực hiện chuyển đổi đặc trưng bằng mô hình Zipformer để trả về chuỗi văn bản tiếng Việt chính xác và alignments từ.

**Why this priority**: Đây là luồng nghiệp vụ cơ bản cốt lõi nhất của hệ thống nhận dạng giọng nói, giúp kiểm chứng mô hình chạy đúng và tương tác thành công với cơ sở dữ liệu.

**Independent Test**: Có thể được kiểm chứng độc lập bằng cách gửi một tệp âm thanh WAV tiêu chuẩn qua công cụ CURL tới endpoint `/api/upload` của Web Backend Gateway, kết quả trả về văn bản nhận dạng tương đương với script `jit_pretrained.py`.

**Acceptance Scenarios**:

1. **Given** Người dùng đã chọn thuật toán "Greedy Search", **When** Người dùng tải lên tệp âm thanh có giọng Bắc, **Then** Hệ thống trả về văn bản nhận dạng tiếng Việt chính xác cùng thuộc tính dialect "NORTHERN" và độ tin cậy tương ứng.
2. **Given** Người dùng đã chọn cấu hình "Modified Beam Search" với beam size = 4, **When** Gửi yêu cầu nhận dạng tệp WAV, **Then** Mô hình Zipformer thực thi giải mã theo chùm và trả về kết quả tối ưu trong thời gian thực.

---

### User Story 2 - Nhận dạng giọng nói trực tuyến thời gian thực (Priority: P2)

Người dùng bật microphone từ giao diện Web và nói trực tiếp, hệ thống liên tục ghi nhận, nén dữ liệu và truyền trực tiếp qua WebSocket để hiển thị văn bản tích lũy tức thì trên màn hình.

**Why this priority**: Nâng cao trải nghiệm người dùng, kiểm thử khả năng xử lý bất đồng bộ, stream dữ liệu liên tục không gây nghẽn luồng (non-blocking).

**Independent Test**: Kết nối WebSocket thông qua `websocat` trực tiếp tới `/api/stream` của Gateway, thực hiện gửi handshake và stream các chunk âm thanh PCM thô để nhận về các bản tin `transcript_update`.

**Acceptance Scenarios**:

1. **Given** Client đã hoàn tất handshake cấu hình streaming, **When** Client gửi liên tục các mảng bytes nhị phân PCM (16kHz Mono 16-bit), **Then** Hệ thống phục vụ duy trì trạng thái phiên và liên tục phản hồi văn bản tiếng Việt tương ứng.
2. **Given** Client gửi tin nhắn "stop", **When** Kết nối vẫn hoạt động, **Then** Hệ thống hoàn tất giải mã phần đệm cuối cùng, trả về kết quả cuối "finished" và đóng kết nối.

---

### Edge Cases

- **Mất kết nối đột ngột**: Khi client ngắt kết nối WebSocket giữa chừng mà chưa gửi frame "stop", hệ thống phục vụ phải dọn dẹp đối tượng giải mã `DecodeStream` tương ứng trong bộ nhớ để tránh rò rỉ RAM.
- **Hết thời gian chờ handshake (Handshake Timeout)**: Nếu client kết nối WebSocket nhưng không truyền đi frame text "handshake" hợp lệ trong vòng 5.0 giây, máy chủ serving phải đóng kết nối với mã code `1008` (Policy Violation).
- **Tham số giải mã vượt biên giới hạn**: Nếu client truyền tham số giải mã (ví dụ: `beam_size = 25` hoặc `method = "unknown"`), máy chủ phải từ chối ngay lập tức bằng lỗi HTTP 400 hoặc đóng kết nối WS tương ứng.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Máy chủ ASR MUST tự động nạp mô hình TorchScript JIT tại đường dẫn cấu hình `viet_iter3_pseudo_label/exp/jit_script.pt` khi khởi động.
- **FR-002**: Máy chủ ASR MUST nạp bảng từ vựng BPE tiếng Việt từ file `viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt` khi khởi động để ánh xạ token ID sang chữ viết.
- **FR-003**: Hệ thống MUST trích xuất đặc trưng Log-Mel Fbank 80 chiều bằng thư viện `torchaudio.compliance.kaldi.fbank` để đảm bảo tính độc lập và không phụ thuộc việc biên dịch thư viện C++ ngoài.
- **FR-004**: Trình phục vụ MUST hỗ trợ trích xuất đặc trưng từ buffer byte của file WAV (cho HTTP REST) lẫn mảng raw int16 PCM (cho WebSocket Streaming).
- **FR-005**: Đối với luồng WebSocket, máy chủ serving MUST quản lý động danh sách các đối tượng `DecodeStream` trong RAM liên kết với từng phiên `session_id` để duy trì trạng thái ẩn (hidden states) giữa các chunk âm thanh.
- **FR-006**: Lớp giải mã MUST tích hợp thuật toán giải mãgreedy search (Transducer loop) và modified beam search kế thừa từ mã nguồn `ASR/zipformer`.

### Key Entities

- **ServingModel**: Thực thể đại diện cho mô hình giải mã Zipformer đã nạp vào RAM dưới dạng TorchScript JIT.
- **SymbolTable**: Bảng ký hiệu từ vựng ánh xạ ID sang chữ viết tiếng Việt.
- **DecodeStream**: Thực thể lưu trữ trạng thái giải mã trực tuyến, bao gồm các tensor trạng thái ẩn của mô hình, số lượng đặc trưng đã đọc, và danh sách các giả thuyết chùm giải mã (hypothesis lists).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Quá trình trích xuất đặc trưng Fbank và forward qua mô hình cho một tệp âm thanh 10 giây phải hoàn thành trong thời gian dưới 1.0 giây trên CPU.
- **SC-002**: Độ trễ cập nhật kết quả từng phần (Time-to-First-Token) của luồng WebSocket streaming phải nhỏ hơn 350 mili giây kể từ khi nhận được chunk âm thanh.
- **SC-003**: Hệ thống phải dọn dẹp thành công 100% đối tượng `DecodeStream` trong vòng 100 mili giây sau khi đóng kết nối WebSocket.

## Assumptions

- Môi trường máy chủ có sẵn các thư viện PyTorch, Torchaudio và k2 tương thích với phiên bản biên dịch của mô hình.
- Mô hình chạy mặc định trên CPU (nếu có GPU sẽ tự động nạp thiết bị `cuda:0`).
- Tệp checkpoint và bảng từ vựng tồn tại đúng vị trí trong thư mục `viet_iter3_pseudo_label`.
