# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk:
- Vector store, embedding model, phiên bản và số chiều:
- Agent model, cách chạy và phân chia dev/test:
- Cách xác định câu có/thiếu evidence và tiêu chí chấm:

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense, top-k=3, candidate-k=20, không filter/rerank, context budget=1200 ký tự | Mốc so sánh; embedding `text-embedding-3-small` (1536 dimensions) | dev | 8/15 (53.33%) | 14/15 (93.33%) | 8/15 (53.33%) | Median 211.591 ms; chi phí chưa tách riêng |
| Phương pháp 2 | BM25, top-k=3, không filter/rerank, context budget=1200 ký tự | So sánh lexical retrieval với Dense | dev | 8/15 (53.33%) | 11/15 (73.33%) | 9/15 (60.00%) | Median 0.557 ms; chi phí chưa tách riêng |
| Phương pháp 3 | Hybrid RRF, candidate-k=20, top-k=3, không filter/rerank, context budget=1200 ký tự | Kết hợp Dense và BM25 bằng Reciprocal Rank Fusion | dev | 8/15 (53.33%) | 13/15 (86.67%) | 8/15 (53.33%) | Median 7.968 ms; chi phí chưa tách riêng |
| Cải thiện 1 | Dense, top-k=5, context budget=1200 ký tự; các thiết lập khác giữ nguyên baseline | Tăng số chunk được lấy để kiểm tra khả năng tăng evidence; kết quả ghi nhận thay đổi không đáng kể, chưa có summary chi tiết trong ZIP | dev | — | — | — | Chưa có số liệu thời gian/chi phí được cung cấp |
| Cải thiện 2 | Dense/BM25/Hybrid RRF, top-k=3, context budget=2000 ký tự; các thiết lập khác giữ nguyên baseline | Tăng context budget để hạn chế trường hợp gold evidence được retrieve nhưng bị loại khỏi context | dev | 8/15 (53.33%) ở cả 3 phương pháp | Dense 14/15 (93.33%); BM25 11/15 (73.33%); Hybrid 13/15 (86.67%) | Dense 12/15 (80.00%); BM25 11/15 (73.33%); Hybrid 12/15 (80.00%) | Dense median khoảng 209.26 ms; thời gian BM25/Hybrid và chi phí chưa có số liệu xác nhận |
| Cấu hình đã chọn | Dense, top-k=3, candidate-k=20, không filter/rerank, context budget=2000 ký tự; `text-embedding-3-small` (1536 dimensions) | Chọn Dense và tăng context budget lên 2000 dựa trên Dev: Context Evidence Rate tăng từ 8/15 lên 12/15 | test | Chưa có kết quả trong ZIP | Chưa có kết quả trong ZIP | Chưa có kết quả trong ZIP | Chưa có số liệu Test cho cấu hình này |
| Baseline | Dense, top-k=3, candidate-k=20, không filter/rerank, context budget=1200 ký tự | Đánh giá baseline trên tập Test; không dùng để điều chỉnh cấu hình | test | 2/5 (40.00%) | 3/5 (60.00%) | 2/5 (40.00%) | Median 320.635 ms; chi phí chưa tách riêng |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:


## Retrieval – kết quả Dev/Test và hai cải thiện

Cấu hình benchmark: cùng corpus/index và bộ câu hỏi; retrieval metrics chỉ tính câu answerable. Baseline dùng Dense, top-k=3, context budget=1,200 ký tự; hai thử nghiệm lần lượt tăng top-k và tăng context budget.

| Phương pháp / cấu hình | Split | n | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median latency |
|---|---|---:|---:|---:|---:|---:|
| Dense, k=3, budget 1200 | Dev | 15 | 8/15 (53.33%) | 14/15 (93.33%) | 8/15 (53.33%) | 211.591 ms |
| BM25, k=3, budget 1200 | Dev | 15 | 8/15 (53.33%) | 11/15 (73.33%) | 9/15 (60.00%) | 0.557 ms |
| Hybrid, k=3, budget 1200 | Dev | 15 | 8/15 (53.33%) | 13/15 (86.67%) | 8/15 (53.33%) | 7.968 ms |
| Dense, k=3, budget 2000 | Dev | 15 | 8/15 (53.33%) | 14/15 (93.33%) | 12/15 (80.00%) | ~209.26 ms |
| BM25, k=3, budget 2000 | Dev | 15 | 8/15 (53.33%) | 11/15 (73.33%) | 11/15 (73.33%) | — |
| Hybrid, k=3, budget 2000 | Dev | 15 | 8/15 (53.33%) | 13/15 (86.67%) | 12/15 (80.00%) | — |
| Dense, k=3, budget 1200 | Test | 5 | 2/5 (40.00%) | 3/5 (60.00%) | 2/5 (40.00%) | — |
| BM25, k=3, budget 1200 | Test | 5 | 4/5 (80.00%) | 4/5 (80.00%) | 4/5 (80.00%) | — |
| Hybrid, k=3, budget 1200 | Test | 5 | 3/5 (60.00%) | 3/5 (60.00%) | 3/5 (60.00%) | — |

**Cải thiện 1 – tăng top-k từ 3 lên 5:** metric retrieval thay đổi không đáng kể; tăng số chunk nhưng không giải quyết rõ vấn đề evidence bị bỏ khỏi context. Không chọn làm cấu hình cuối.

**Cải thiện 2 – tăng context budget từ 1,200 lên 2,000:** Context Evidence Rate trên Dev tăng Dense 53.33% → 80.00%, BM25 60.00% → 73.33%, Hybrid 53.33% → 80.00%; Hit@1 và Evidence Hit@3 không đổi. Chọn Dense, k=3, budget=2,000 để ưu tiên giữ evidence trong context cuối.

**Hai lỗi Dev đã phân tích:**
- `dev-001`: gold chunk `nd13-2023-p2-legal-0001` có trong kết quả truy xuất ở rank 3 nhưng bị loại khỏi context khi budget 1,200.
- `dev-013`: gold chunk `nd13-2023-p4-legal-0001` có trong kết quả truy xuất ở rank 3 nhưng bị loại khỏi context khi budget 1,200.
- Thử tăng budget lên 2,000; Context Evidence Rate tăng, cho thấy nguyên nhân chính ở bước chọn/cắt context thay vì hoàn toàn không retrieve được evidence.

**Cấu hình đóng băng trước Test:** Dense, top-k=3, context budget=2,000; embedding `text-embedding-3-small` (1536 chiều); Agent strategy B, model `gpt-6-luna`, tối đa 4 lượt search. Không dùng Test để tinh chỉnh cấu hình.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:


## Kết quả audit Agent trên Test

Benchmark Test có 5 câu answerable và 2 câu unanswerable. Đánh giá Grounded answer theo thang 0–2: 0 = từ chối nhầm/sai hoặc không có căn cứ; 1 = đúng một phần; 2 = trả lời đúng và có evidence hỗ trợ.

| Cấu hình / split | Tool use | Số lượt search | Grounded answer TB | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline / Test | 5/5 (100%) | 12 | 1.6/2 | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20%) |
| Selected / Test | 5/5 (100%) | 11 | 1.6/2 | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20%) |

| Cấu hình | Avg latency / query | Tổng token trong trace |
|---|---:|---:|
| Baseline | 6,176.79 ms | 5,254 |
| Selected | 5,517.94 ms | 6,847 |

**Nhận xét chính:** Cả hai cấu hình trả lời đúng 4/5 câu có thể trả lời và từ chối đúng 2/2 câu không có evidence. Cả hai đều từ chối nhầm `test-001`. Selected giảm thời gian trung bình khoảng 10.7% nhưng tổng token trong trace tăng do nội dung context/prompt lớn hơn. Với 7 câu Test, chênh lệch này chỉ mang tính mô tả, chưa đủ để kết luận cấu hình selected tốt hơn về chất lượng.

### Audit theo query

| Query | Loại | Baseline | Selected | Nhận xét |
|---|---|---|---|---|
| test-001 | Answerable | Từ chối (0) | Từ chối (0) | False refusal. Agent tìm thấy chunk trang 2 nhưng context không chứa đoạn liệt kê dữ liệu vị trí; các lượt tìm tiếp theo vẫn trả về chunk không liên quan. |
| test-002 | Answerable | Đúng (2), citation hợp lệ | Đúng (2), citation hợp lệ | Chunk `nd13-2023-p10-legal-0001` nêu rõ bên kiểm soát/bên kiểm soát và xử lý chịu trách nhiệm chứng minh sự đồng ý. |
| test-003 | Answerable | Đúng (2), citation hợp lệ | Đúng (2), citation hợp lệ | Chunk `nd13-2023-p15-legal-0003` hỗ trợ trường hợp khẩn cấp để bảo vệ tính mạng/sức khỏe. |
| test-004 | Answerable | Đúng (2), citation hợp lệ | Đúng (2), citation hợp lệ | Chunk `nd13-2023-p15-legal-0001` hỗ trợ nghĩa vụ xóa dữ liệu không thể khôi phục khi mục đích xử lý hoàn thành. |
| test-005 | Answerable | Đúng (2), citation hợp lệ | Đúng (2), citation hợp lệ | Chunk `nd13-2023-p5-legal-0003` liệt kê kỷ luật, xử phạt hành chính hoặc xử lý hình sự. |
| test-006 | Unanswerable | Từ chối đúng | Từ chối đúng | Tài liệu không quy định AES-256 hay RSA-2048; không bịa câu trả lời. |
| test-007 | Unanswerable | Từ chối đúng | Từ chối đúng | Tài liệu không nêu thời hạn lưu log tối thiểu; không bịa câu trả lời. |

### Nhận định và giới hạn

- Điểm yếu nổi bật không nằm ở việc bịa citation mà ở khả năng truy xuất đủ evidence cho `test-001`: đây là câu answerable nhưng Agent dừng ở trạng thái abstain sau 4 lượt search.
- Cả hai cấu hình có citation validity 4/4; các citation được trích đều có mặt trong context cuối và hỗ trợ nội dung trả lời.
- Cần ưu tiên sửa chunking/index hoặc truy vấn cho nhóm câu hỏi phân loại dữ liệu ở Điều 2; tăng số lượt search đơn thuần chưa giải quyết được lỗi này.
- Số liệu chi phí trong ảnh [cost.png](cost.png) và [usage.png](usage.png): 122,257 total tokens và $0.03 total spend. 
- Trace riêng cho Test ghi nhận 5,254 tokens (baseline) và 6,847 tokens (selected). Không suy ra chi phí USD từ token vì thiếu breakdown giá theo model/cached input/output trong trace.
