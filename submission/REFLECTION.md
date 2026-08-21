# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm em bất ngờ nhất không phải là việc fine-tune có thắng prompt baseline hay không, mà là **ranh giới hiệu năng quá gắt gao**. Em từng nghĩ model fine-tune sẽ có vài ca thua baseline `(b)` ở các ticket phức tạp, nhưng thực tế khi soi kỹ 9 ticket khó nhất thì `(b)` không thắng được `(c)` ở bất kỳ ticket nào (toàn hoà hoặc FT thắng). 

Thế nhưng, vừa bước chân ra khỏi miền dữ liệu huấn luyện (tập regression) thì model lại sụp đổ hoàn toàn. Điển hình như câu hỏi cơ bản *"1 km bằng bao nhiêu mét?"*, thay vì đáp "1000m" như model gốc thì bản fine-tune lại cố ép về một khối JSON triage rỗng với `product: null`. Model không hề "giảm độ thông minh từ từ", mà là làm cực kỳ xuất sắc trong đúng cái hộp được dạy và "ngáo" hẳn khi ra ngoài cái hộp đó.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Về thời gian máy chạy, đúng là NB4 (3 run đối chứng `attn_only`, `wrong_lr`, `qlora`) ngốn nhiều nhất — chiếm hơn 51 phút trên tổng gần 90 phút toàn pipeline (chiếm ~57.6%).

Tuy nhiên, phần ngốn nhiều thời gian và công sức của em nhất thực tế lại là **khâu đối soát và phân tích số liệu sau khi train**. Việc ngồi soi từng file log, bóc tách JSON để tìm nguyên nhân tại sao loss thấp mà eval không tăng, hay viết thêm script để so sánh output từng ticket giữa các baseline tốn nhiều nơ-ron hơn nhiều so với việc chỉ ngồi chờ Colab chạy.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước đây em luôn mặc định trong đầu: *"Train loss càng ép xuống thấp thì model ra lò càng xịn"*.

Sau bài lab này thì số liệu thực tế đã chứng minh điều ngược lại. Cụ thể, run `attn_only` đạt train loss thấp hơn hẳn bản chuẩn `correct` (0.5371 so với 0.6263), nhưng khi đưa vào đánh giá tác vụ thực tế thì điểm số hai bên chỉ ngang ngửa nhau. Nếu chỉ nhìn vào loss trên đồ thị thì rất dễ bị đánh lừa là attention-only tốt hơn, trong khi bản chất nó chỉ đang học vẹt/overfit nhanh hơn trên tập train.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Em dùng AI để hỗ trợ tăng tốc công việc: viết các hàm Python gom và parse log từ các file JSON (`mask_proof`, `baselines_frozen`, `runs.csv`, `autopsy`,...), hỗ trợ format bảng biểu và dựng khung sườn cho báo cáo.

Chỗ AI làm sai rõ nhất là tính **"đoán mò thay vì đo thật"**: Khi phân tích mục so sánh các ticket khó, AI từng tự tiện phán đoán rằng các ticket điểm thấp của fine-tune là do "FT bị baseline (b) vượt mặt". Đến khi em chạy code sinh kết quả và eval trực tiếp thì thực tế baseline (b) còn tệ hơn. Bài học rút ra là tuyệt đối không tin các nhận định định tính của AI nếu chưa có số liệu đo đạc thực tế từ code.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Bước đầu tiên của em không phải là nhảy vào chọn model hay config tham số, mà là **ngồi lại với khách hàng để chốt bộ test benchmark và phạm vi sử dụng**.

Em sẽ hỏi rõ: *"Ngoài việc giải quyết đúng bài toán này, model có cần trả lời các câu hỏi hội thoại thông thường nữa không?"*. Đồng thời, em sẽ chuẩn bị sẵn một tập validation đa nhiệm (regression set) và trộn ngay khoảng 2-5% dữ liệu tổng quát vào tập train từ đầu. Lab này cho thấy hiện tượng quên tri thức cũ (catastrophic forgetting) là điều chắc chắn sẽ xảy ra nếu chỉ huấn luyện dữ liệu đơn ngành một chiều.
