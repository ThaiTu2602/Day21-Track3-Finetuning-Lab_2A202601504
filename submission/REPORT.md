# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thái Tú  **MSSV**: 2A202601504  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (fp16, sm_75)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage (corpus mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 steps |

**Template có giữ khối `<think>` không?** Có — `results/template_check.json` trả về
verdict `"reasoning preserved — safe to train on traces"`. Chat template của Qwen3.5
bảo toàn nguyên vẹn khối `<think>...</think>` khi `apply_chat_template`, nên không có
hiện tượng âm thầm xoá reasoning trace trước khi tính loss.

Về `max_length`: p95 đo được chỉ 98 token (`token_stats.json` gợi ý
`suggested_max_length=256`), nhưng tier T4 mặc định đặt 1024 và tôi giữ nguyên. Lý do:
(1) `batch_size=1` + `gradient_accumulation_steps=16` + `gradient_checkpointing=True`
nên VRAM còn dư nhiều (peak đo được ~12 GB / 14.6 GB khả dụng) — hạ `max_length` không
tiết kiệm được gì đáng kể; (2) không mẫu nào trong 250 mẫu vượt quá 101 token (`max`
đo được), nên phần dư 1024-101 chỉ là padding không dùng tới; (3) 1024 tạo biên an toàn
cho các ticket dài bất thường ngoài phân phối huấn luyện, phòng trường hợp bị cắt cụt
(truncate) làm hỏng nhãn JSON ở cuối chuỗi.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Chỉ phần phản hồi của assistant (JSON) và token kết thúc `<|im_end|>` nằm trong vùng
được tính loss (`supervised`, 41.49% tổng token của ví dụ này; tổng hợp trên cả 225 mẫu
huấn luyện, NB3 log cho con số tương đương: 9014/20951 token = 43.0%). Toàn bộ system
prompt, câu hỏi của người dùng và thẻ `<think>` rỗng đều bị gán `-100` (masked). Điều
này đảm bảo model học sinh ra câu trả lời đúng định dạng, thay vì học lặp lại câu hỏi
— khớp với yêu cầu 1.1 của rubric (`supervised_fraction` phải < 0.95).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3394.3 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1067.3 |
| (c) LoRA fine-tune | 0.970 | 0.4556 | 1.000 | 1576.8 |

**(b) có thật sự mạnh hơn (a) không?** Có, rõ ràng: target đi từ 0.000 lên 0.765, format
từ 0.000 lên 1.000 (model bắt đầu trả JSON hợp lệ), và latency giảm ~3.2× (3394ms →
1067ms) vì output ngắn gọn hơn, không lan man. `regression` giống hệt nhau giữa (a) và
(b) (0.7578) vì cả hai đều là base model chưa fine-tune, chỉ khác system prompt cho tác
vụ target — probe regression dùng `system=None` nên không bị ảnh hưởng.

**Tôi có sửa `OPTIMIZED_PROMPT` không?** Không. `git diff` trên `src/labkit/generate.py`
so với gốc repo không có gì khác biệt — tôi dùng nguyên bản `OPTIMIZED_PROMPT` mặc định
của lab. `optimized_prompt_sha` trong `results/baselines_frozen.json` (`719e74d3b6232053`)
sẽ được `make verify` tự đối chiếu với hash tính từ `labkit.generate.OPTIMIZED_PROMPT`
để xác nhận điều này khi tôi chạy verify trong Colab.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 (10×) | 0.6263 | 0.970 | 1055.3 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 (10×) | 0.5371 | 0.970 | 855.8 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 (1×) | 1.5704 | 0.000 | 1003.2 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 (10×) | 0.7058 | 0.940 | 1066.5 | 7.09 |

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct` (32,456,704 vs
32,464,896, lệch 0.025% — đạt yêu cầu <5%). Trên tập target nó thắng, thua, hay hoà?**
Hoà tuyệt đối: cả hai đạt target = 0.970. Nhưng thứ tự đó **không** giống thứ tự theo
train loss của NB4: ở đó `attn_only` có loss thấp hơn (0.5371 so với 0.6263 của
`correct`), nên nếu chỉ nhìn cột loss, tôi sẽ kết luận sai rằng `attn_only` "thắng". Trên
thang đo thật (target task) thì hai cấu hình cho kết quả như nhau. Điều này nói rằng ở
ngân sách tham số bằng nhau, *rank* càng cao (283 so với 16) không tạo ra lợi thế thật
sự trên tác vụ — nó chỉ giúp ép loss huấn luyện xuống thấp hơn, nhiều khả năng là do
ghi nhớ cục bộ trên một tập nhỏ (225 mẫu, 30 step) chứ không phải khái quát hoá tốt hơn.
Vị trí gắn adapter (bao phủ 12 loại module ở `text-linear` so với chỉ 2 loại q/v) mới là
yếu tố quyết định trần năng lực đạt được — `attn_only` cần rank gấp 17.7 lần chỉ để *hoà*
chứ không vượt qua được `correct`. Đây đúng là Lỗi #3 mà rubric nhắc: chấm bằng chỉ số
thay thế (loss) thay vì bằng năng lực trên tác vụ (target) sẽ dẫn tới kết luận sai.

**4.2 — `wrong_lr` chỉ khác đúng một con số (LR 1e-5 thay vì 1e-4, tức thang full-FT
thay vì thang LoRA ×10). Đường loss khác nhau ra sao?** Ở step đầu cả bốn run xuất phát
từ cùng loss (2.163, cùng init), nhưng trong khi `correct`/`attn_only`/`qlora` giảm rất
nhanh xuống 0.02–0.04 ở cuối, `wrong_lr` giảm rất chậm và dừng ở loss trung bình 1.57
(loss cuối mỗi log-step ~1.1). Nếu chỉ nhìn đường loss mà không biết LR, tôi sẽ kết luận
sai rằng model "đang học chậm, cần thêm step" — trong khi thực tế `target=0.000` và
`format=0.000`: model không tạo ra được bất kỳ JSON hợp lệ nào, tức là hoàn toàn chưa
học được tác vụ. Nguyên nhân là LoRA khởi tạo ma trận A/B gần như đóng góp bằng 0 vào
forward pass ban đầu, nên cần LR cao hơn full fine-tune (theo *LoRA Without Regret*,
thường ×10) để di chuyển đủ xa trong một ngân sách step nhỏ (30 step); dùng LR thang
full-FT khiến adapter gần như đứng yên.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?** Tiết kiệm 12.01 − 7.09 =
4.92 GB, tức ~41% VRAM peak — khớp với mức tiết kiệm lý thuyết của nạp 4-bit. Cái giá
phải trả: target giảm nhẹ từ 0.970 xuống 0.940 (Δ = −0.03, tức khoảng 1.5/50 item), thời
gian train gần như không đổi (1066.5s so với 1055.3s của `correct`, chậm hơn <1%), và
format vẫn giữ nguyên 1.000. Số đo của tôi **không ủng hộ mạnh** khuyến nghị "không dùng
QLoRA cho dòng model này" (deck §12): mức giảm target 0.03 trên 50 item nằm trong biên
độ nhiễu hợp lý, trong khi phần thưởng VRAM (41%) là rất lớn và có thể quyết định việc
có train được hay không trên GPU nhỏ hơn T4. Nếu ràng buộc chính là VRAM, QLoRA ở đây là
một đánh đổi hợp lý chứ không phải điều nên tránh tuyệt đối — dù cỡ mẫu 50 item chưa đủ
để khẳng định chắc chắn không có suy giảm ở quy mô lớn hơn.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.302` · `valid_trace_rate = 0.00`

Diễn giải: Cổng FAILED vì tuy fine-tune vượt xa baseline (b) trên tác vụ mục tiêu
(target 0.765 → 0.970, format đã đạt 1.000, latency vẫn thấp), nó lại phá huỷ nghiêm
trọng năng lực tổng quát của model — `regression` sụp từ 0.7578 xuống 0.4556, tức giảm
0.302, vượt xa ngưỡng cho phép chỉ 0.020 (gấp 15 lần). `valid_trace_rate = 0.00` cho
thấy trên tập regression (câu hỏi kiến thức chung, không liên quan triage), model
fine-tune không còn tạo ra khối `<think>` hợp lệ nào nữa — dấu hiệu kinh điển của việc
model đã học "bất kỳ input nào cũng phải trả về JSON triage", đúng như deck §14.3 mô tả
về catastrophic forgetting khi train trên một phân phối quá hẹp (250 ticket → JSON,
không có dữ liệu tổng quát nào xen kẽ). Nói cách khác, adapter `correct` đạt điểm cao
trên đúng tác vụ nó được huấn luyện, nhưng không an toàn để triển khai như một trợ lý đa
năng vì nó sẽ trả JSON triage cho cả câu hỏi không liên quan. Hướng khắc phục đúng theo
rubric là trộn 1–5% dữ liệu tổng quát (replay data) vào tập huấn luyện rồi đo lại
`regression`, chứ không phải nới lỏng ngưỡng 0.020.

---

## 6. Định tính — bắt buộc có cả ca THUA

**Phát hiện quan trọng, không cherry-pick:** tôi đã sinh dự đoán thật của `(b)` trên cả
9 item — 6 item mà `(c)` không đạt điểm tuyệt đối (i=3,5,12,39,41,46, đều =0.75) và 3
item `(c)` đạt 1.0 (i=47,48,49) — rồi so từng trường (intent/urgency/product/sentiment)
với nhãn gốc. Kết quả: **`(b)` không thắng `(c)` ở bất kỳ item nào trong cả hai nhóm.**
Vì `results/qualitative.json` cho thấy `(c)` không có item nào dưới 0.75 trên toàn bộ 50
item, và ở đúng 6 item 0.75 đó `(b)` chưa từng vượt qua, nên đây là bằng chứng đầy đủ
trên cả tập target: **không tồn tại item nào mà prompt "tối ưu" (b) đánh bại fine-tune
(c)** ở tác vụ triage.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | i=49 — "...ốp lưng điện thoại mã đơn VN833689. Sai màu..." | san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh | san_pham_loi / **cao**✗ / ốp lưng điện thoại / **tieu_cuc**✗ (0.50) | đúng cả 4 trường (1.0) | ✅ **FT thắng** — (b) sai cả urgency lẫn sentiment |
| 2 | i=5 — "...nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện." | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | **hoan_tien**✗ / **cao**✗ / nồi chiên không dầu / trung_tinh (0.50) | san_pham_loi / **trung_binh**✗ / nồi chiên không dầu / trung_tinh (0.75) | ✅ **FT thắng** — (b) sai cả intent lẫn urgency |
| 3 | i=41 — "...đèn bàn LED mã đơn OD436045. Giao hàng chậm..." | van_chuyen / thap / đèn bàn LED / tich_cuc | **hoi_thong_tin**✗ / **trung_binh**✗ / đèn bàn LED / tich_cuc (0.50) | van_chuyen / **trung_binh**✗ / đèn bàn LED / tich_cuc (0.75) | ✅ **FT thắng** — (b) sai cả intent lẫn urgency |
| 4 | i=3 — "...bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền." | hoan_tien / **thap** / bình giữ nhiệt / tich_cuc | hoan_tien / **trung_binh**✗ / bình giữ nhiệt / tich_cuc (0.75) | hoan_tien / **trung_binh**✗ / bình giữ nhiệt / tich_cuc (0.75) | 🟰 **Hoà** — cả hai cùng sai đúng 1 trường, cùng giá trị sai |
| 5 | i=12 — "...áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện." | san_pham_loi / **thap** / áo khoác gió / tich_cuc | san_pham_loi / **trung_binh**✗ / áo khoác gió / tich_cuc (0.75) | san_pham_loi / **trung_binh**✗ / áo khoác gió / tich_cuc (0.75) | 🟰 **Hoà** — cùng sai `urgency`, cùng giá trị |

Có mẫu chung nào ở các ca "thua/hoà" của (b) không? Có — **mọi trường sai của cả hai mô
hình đều rơi vào `urgency`**, và cụ thể là các ticket có cụm mơ hồ về thời gian ("Chưa
thấy tiền", "Khi nào tiện", "Bị lỗi") thay vì từ khoá rõ ràng như "gấp"/"ngay". Điều này
cho thấy `urgency` là trường khó nhất trong schema đối với **cả hai** phương pháp — đây
là giới hạn của bản thân dữ liệu/nhãn (ambiguity), không phải điểm yếu riêng của
fine-tune. Ở những item mà `(b)` sai *thêm* cả `intent` (i=5, 41, 49), `(c)` vẫn đúng
`intent` — cho thấy fine-tune khái quát hoá nhãn `intent` tốt hơn hẳn một prompt dài dựa
trên few-shot/mô tả, dù cả hai học từ cùng một base model.

**Vậy "FT thua" ở đâu?** Không phải ở tập target — ở tập **regression** (câu hỏi chung,
ngoài miền triage), đúng như `verdict.json` đã chỉ ra (regression 0.7578 → 0.4556,
`valid_trace_rate=0.00`). Đây mới là ca thua thật, và nó không phải cherry-pick vì nó
chính là lý do cổng hồi quy FAILED:

| # | Câu hỏi (regression, ngoài miền) | (b) base + optimized prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|
| 6 | "1 km bằng bao nhiêu mét?" | "...1 kilômét tương đương với 1000 mét. **1 km = 1000 m**." (đúng, rõ ràng) | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh", "intent_confidence": 0.95, ...}` — **không có câu trả lời "1000 mét" ở đâu cả** | ❌ **FT thua nặng** — câu trả lời thật (1000 mét) biến mất hoàn toàn, model chỉ nhả ra khung JSON triage rỗng với `product: null` |
| 7 | "Viết một câu chúc mừng sinh nhật bằng tiếng Việt." | "Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe..." (tự nhiên, đúng yêu cầu) | `{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "tone": "trung_thinh", "content": "Chúc bạn..."}` | ❌ **FT thua** — nội dung câu chúc vẫn đúng nhưng bị nhồi vào một schema JSON tự bịa; đáng chú ý `"urgency": "trung_tinh"` mượn nhầm từ vựng của trường `sentiment` trong lúc train — bằng chứng model đã lẫn lộn/ghi nhớ từ vựng của schema hẹp thay vì hiểu ngữ cảnh |

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi sẽ **không** deploy bản fine-tune `adapters/correct` ở dạng hiện tại.
Trên đúng tác vụ triage, nó thắng rõ ràng cả hai baseline (target 0.970, vượt cả prompt
"tối ưu" 0.765 và gấp đôi so với prompt "naive"), và format đạt tuyệt đối 1.000 — nghĩa
là bài toán này *xứng đáng* được fine-tune, không phải trường hợp "prompt engineering là
đủ". Nhưng nó fail cổng hồi quy vì catastrophic forgetting: regression giảm 0.302, 15 lần
ngưỡng cho phép, và `valid_trace_rate=0.00` cho thấy khả năng suy luận tổng quát gần như
biến mất. Đòn bẩy thật sự trong lab này, theo thứ tự quan trọng đo được: (1) **learning
rate đúng thang** — thiếu nó (`wrong_lr`) khiến model không học được gì (target=0), đây
là điều kiện cần trước tiên; (2) **vị trí gắn adapter** (bao phủ toàn bộ lớp linear thay
vì chỉ q/v) quan trọng hơn *rank* — `attn_only` cần rank gấp 17.7× chỉ để hoà chứ không
thắng `correct`; (3) **chất lượng/độ đa dạng của dữ liệu huấn luyện** mới là nguyên nhân
gốc của FAILED — 250 mẫu toàn bộ đều là "ticket → JSON" không có dữ liệu tổng quát xen
kẽ, nên model học sai quy luật "mọi input đều cần JSON". Mask (`assistant-only`,
supervised_fraction=0.41) đã đúng ngay từ đầu nên không phải nguyên nhân. Vì vậy bước
tiếp theo bắt buộc trước khi deploy là trộn 1–5% dữ liệu hội thoại tổng quát vào tập
train rồi đo lại `regression`, không phải là hạ ngưỡng chấp nhận.

**Ba điều tôi học được:**
1. Loss huấn luyện có thể xếp hạng **ngược** so với năng lực thật trên tác vụ: `attn_only`
   có loss thấp hơn `correct` (0.5371 vs 0.6263) nhưng trên target hai bên chỉ hoà tuyệt
   đối (0.970 = 0.970) — nếu chỉ nhìn loss, tôi đã kết luận nhầm rằng rank cao ở vị trí
   hẹp là lựa chọn tốt hơn.
2. Đạt điểm target cao không đồng nghĩa với an toàn để deploy. Fine-tune của tôi không hề
   thua baseline (b) ở bất kỳ ticket nào trong tập target (đã kiểm chứng từng trường trên
   9 ticket khó nhất), nhưng vẫn FAILED vì regression sụp 0.302 — hai câu hỏi "nó có giỏi
   việc được giao không" và "nó có còn an toàn để dùng chung không" là hai câu hỏi khác
   nhau, và chỉ đo câu đầu thì sẽ bỏ sót thất bại thật sự.
3. Một sai lệch tưởng rất nhỏ (LR thiếu đúng một số 0: 1e-5 thay vì 1e-4) làm model không
   học được gì (`wrong_lr`: target=0.000), trong khi một thay đổi tưởng lớn (chuyển sang
   4-bit QLoRA) chỉ làm giảm 0.03 điểm target — trực giác "thay đổi lớn về kỹ thuật = ảnh
   hưởng lớn về kết quả" là sai; phải đo, không đoán.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu hội thoại tổng quát vào tập
train (theo đề xuất deck §14.3) rồi train lại `correct`, đo lại cổng hồi quy để xem liệu
`regression` có về lại gần 0.7578 mà vẫn giữ được `target` ở mức ~0.97 hay không — đó mới
là câu hỏi thật sự cần trả lời trước khi cân nhắc deploy.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
