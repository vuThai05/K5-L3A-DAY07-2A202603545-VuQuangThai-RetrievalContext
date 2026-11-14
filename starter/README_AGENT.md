# Day 7 – Bước 4: AI Agent

## Cài đặt
Từ thư mục project Day 7:
```powershell
pip install -r starter/requirements.txt
```
Nếu `requirements.txt` hiện tại nằm trong `starter/`, dùng đúng đường dẫn tương ứng. Cần có `openai`, `python-dotenv`, `psycopg[binary]`, `pgvector`, `rank-bm25`.

Trong `starter/.env`:
```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-6-luna
DATABASE_URL=...
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```
Không commit `.env`.

## Chuẩn bị benchmark Agent
Giữ `questions.json` chỉ gồm 20 câu answerable cho Retrieval Evaluation. Ghép 4 câu unanswerable vào file riêng:
```powershell
python starter/prepare_agent_data.py --dev path\to\dev_questions.json --test path\to\test_questions.json --output starter/data/converted/questions_agent.json
```
Hai file đầu vào là benchmark gốc (17 câu dev, 7 câu test). File đầu ra phải có 24 câu: Dev 17, Test 7. Nếu file gốc đang ở thư mục khác, thay đường dẫn tương ứng.

## Chạy Agent trên Dev
```powershell
python starter/run_agent.py --questions starter/data/converted/questions_agent.json --split dev --strategy B --retriever dense --top-k 3 --candidate-k 20 --max-searches 4 --context-budget 1200
```
Thử BM25 và Hybrid trên Dev bằng cách đổi `--retriever bm25` hoặc `--retriever hybrid`. Không cần chạy cả ba trên Test; chọn cấu hình theo Dev rồi chốt trước Test.

## Chạy Test sau khi chốt
```powershell
python starter/run_agent.py --questions starter/data/converted/questions_agent.json --split test --strategy B --retriever dense --top-k 3 --candidate-k 20 --max-searches 4 --context-budget 1200
```

## Output
`starter/results/agent/{dev|test}/{dense|bm25|hybrid}/traces.json` và `summary.json`.
Mỗi trace ghi câu hỏi, status, answer, citations, từng search, context, model, số lượt LLM và latency. API usage token được lưu nếu SDK trả về.

## Audit thủ công
Đọc tối thiểu 5 câu test answerable và 2 câu unanswerable theo BENCHMARK.md. Tạo `reviews.json` với một dòng mỗi query_id:
- Answerable: `answer_grade` = 0/1/2; `citation_support` có một boolean cho mỗi citation.
- Unanswerable: `abstention_correct` = true/false; `citation_support` = [].
Không điền nhãn bằng model. Sau đó dùng `agent_metrics.py` hiện có để tổng hợp theo README/CLI.

## Lưu ý
- Không đưa `expected_answer`, `gold`, `absence_reason` hay nhãn audit vào prompt/model.
- `gpt-6-luna` được đặt làm mặc định theo cấu hình người dùng cung cấp; nếu API báo model not found/unsupported, kiểm tra model ID và quyền truy cập của project API.
- Agent chỉ kiểm tra citation ID có trong context; citation support/grounding phải audit thủ công.
- Đây là bản tích hợp tối thiểu dựa trên `run_retrieval.py` và interface `agentic.py`; chưa xác nhận kết nối tới DB/API của máy local. Không dùng kết quả giả.
