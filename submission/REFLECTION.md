# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Không phải việc fine-tune thắng hay thua, mà là **cách nó thắng/thua**. Tôi nghĩ khi so
từng ticket một, bản fine-tune sẽ có vài ca thua rõ ràng trước baseline (b) — nhưng khi
so thật từng trường trên 9 ticket khó nhất, `(b)` không thắng `(c)` ở bất kỳ ticket nào
trong cả tập target: chỉ có hoà hoặc fine-tune thắng. Cùng lúc đó, ở tập regression
(câu hỏi ngoài miền), fine-tune lại thua thảm — ví dụ rõ nhất là câu "1 km bằng bao
nhiêu mét?", model không trả lời "1000 mét" nữa mà chỉ nhả ra một khối JSON triage rỗng
với `product: null`. Tức là fine-tune không "kém đi một chút ở mọi nơi" như tôi tưởng,
mà là "hoàn hảo trong đúng cái hộp nó được huấn luyện, và sụp hoàn toàn ngay khi ra khỏi
hộp đó". Ranh giới đó sắc hơn tôi nghĩ rất nhiều.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Theo `stage timings` đo được: NB4 (ba run đối chứng: `attn_only`, `wrong_lr`, `qlora`)
chiếm 3104s / 5389s tổng — tức 57.6% tổng thời gian toàn pipeline, hơn cả bốn notebook
còn lại cộng lại. _(Bạn tự đánh giá: điều này có khớp với cảm giác lúc chờ Colab chạy
không, hay phần bạn cảm thấy "mất thời gian" thực ra là lúc debug/đọc hiểu số liệu chứ
không phải lúc máy đang train?)_

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

_(Câu này chỉ bạn trả lời được — đây là một gợi ý có căn cứ từ chính số liệu của bạn,
không phải suy đoán về bạn:)_ Một niềm tin phổ biến là "loss huấn luyện thấp hơn = model
tốt hơn". Số liệu của chính bạn bác bỏ điều đó trực tiếp: `attn_only` có loss thấp hơn
`correct` (0.5371 so với 0.6263) nhưng trên tác vụ thật hai bên chỉ hoà — nếu chọn model
theo loss, bạn sẽ không sai, nhưng nếu tưởng loss thấp hơn nghĩa là "tốt hơn", bạn sẽ kết
luận nhầm rằng attention-only là placement ưu việt hơn.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi (Claude) được dùng để: đọc và tổng hợp log Colab của cả 5 notebook, tính toán số
liệu chéo từ nhiều file JSON (`mask_proof`, `baselines_frozen`, `runs.csv`, `autopsy`,
`verdict`, `qualitative`), soạn khung `REPORT.md` bám theo rubric, và viết các đoạn code
Python để lấy thêm dữ liệu mà pipeline gốc không lưu sẵn (prediction của baseline (b)
theo từng ticket, và ví dụ trên tập regression).

Chỗ nó sai rõ ràng nhất: ở bản nháp đầu tiên của mục 6, tôi (Claude) đã suy đoán — không
có bằng chứng — rằng 3 ticket điểm thấp nhất của fine-tune (i=3, 5, 12) "khả năng là ca
FT thua". Sau khi thật sự sinh dự đoán của `(b)` trên các ticket đó, sự thật ngược lại
hoàn toàn: `(b)` không hề thắng, chỉ hoà hoặc thua. Tôi phải viết lại toàn bộ mục 6. Bài
học: đừng để AI suy đoán số liệu thay vì đo — kể cả khi suy đoán đó "nghe có lý", nó vẫn
cần được xác minh bằng dữ liệu thật trước khi đưa vào report.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Trước khi chạy bất kỳ dòng code
huấn luyện nào, tôi sẽ hỏi khách hàng: "Ngoài tác vụ này, model còn phải trả lời được
những loại câu hỏi nào khác?" — rồi thu thập ngay 1-5% dữ liệu tổng quát/đa dạng để trộn
vào tập train từ đầu, thay vì coi đó là bước "sửa sau nếu regression tụt". Lab này cho
thấy catastrophic forgetting không phải rủi ro nhỏ cần theo dõi — nó là kết quả mặc định
nếu dữ liệu huấn luyện chỉ có một loại input duy nhất.
