# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường

> Track 4 · Ngày 10 · *Nhãn, auto-labeling và Vision–Language Model* · **bài cá nhân** · ≈ 4 giờ trên **Google Colab T4 (miễn phí)**
> Bạn **tự viết** phần huấn luyện và suy luận theo pseudo code trong [`GUIDE.md`](GUIDE.md). Công cụ chia dữ liệu và chấm điểm do giảng viên viết sẵn và **không được sửa**.

| File | Dùng để làm gì |
|---|---|
| `README.md` (file này) | Bức tranh toàn lab, dữ liệu, môi trường, cách nộp |
| [`GUIDE.md`](GUIDE.md) | Từng bước + pseudo code + câu hỏi |
| [`RUBRIC.md`](RUBRIC.md) | Thang điểm 100 (+10) |
| [`REPORT.md`](REPORT.md) | Mẫu báo cáo 10 câu |
| `lab_day10.ipynb` | Khung notebook để bạn điền mã |
| `configs/e0…e5.yaml` | Siêu tham số từng thí nghiệm |
| `scripts/prepare_data.py` | Tải dữ liệu, chia split |
| `scripts/score.py` | **Trọng tài**: chấm `predictions.jsonl` |
| `scripts/check_submission.py` | Kiểm tra bài trước khi nộp |

---

## 1. Mục tiêu học tập

Sau bài lab, bạn có thể:

1. Mô tả được **ViT → projector → LLM** của Qwen3-VL-4B, và nói rõ khối nào bạn đã cập nhật trong từng thí nghiệm.
2. Finetune VLM bằng **QLoRA** và **so sánh bằng số** các cách đóng băng/mở khối (LoRA LLM; thêm projector; thêm ViT).
3. Giải thích vì sao phải chia dữ liệu **theo phiên quay** (không chia ngẫu nhiên) và vì sao **lệch thang toạ độ 0–1000** làm hỏng kết quả.
4. Đọc F1 **kèm khoảng tin cậy** trên một tập test nhỏ và biết kết luận nào còn đứng vững.
5. (Tuỳ chọn) Viết hàm **reward IoU** cho RL với phần thưởng kiểm chứng được.

## 2. Bài toán

Ảnh dashcam 640×360 → model trả **một JSON**:

```json
[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}]
```

Toạ độ là **số nguyên trên thang 0–1000** (tương đối so với chiều rộng/cao ảnh), đúng quy ước grounding của Qwen3-VL trong slide Day 10. Ba lớp: `pothole` (ổ gà), `crack` (nứt), `manhole` (nắp cống). Mỗi ảnh 640×360 chiếm **220 token** vision.

## 3. Dữ liệu

Kaggle [`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes), giấy phép **MIT**.

| | |
|---|---|
| Ảnh | 2.009 ảnh JPG 640×360 (dashcam, quay tháng 2–3/2025) |
| Box | 4.737 (crack 2.519 · pothole 1.261 · manhole 957), trung bình 2,4 box/ảnh |
| Độ khó | Box nhỏ: diện tích trung vị chỉ ≈ 0,7% khung hình |
| Nhãn dùng | `labels-YOLO/` (`class x_center y_center w h`, chuẩn hoá [0,1]) |

**Các ảnh là frame cắt từ video**, nên hai frame liền kề gần như trùng nhau. Nếu chia ngẫu nhiên, ảnh gần giống nhau sẽ nằm ở cả train và test (**rò rỉ**). `prepare_data.py` gom ảnh theo **phiên quay** (ngày + giờ trong tên file) và gán **cả phiên** vào một split. Split này **cố định**: ai chạy cũng ra cùng kết quả.

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Lưu ý: split theo phiên nên phân bố lớp **không đều** (val có rất ít pothole). Đó là tình huống thật, và là lý do bạn phải đọc **F1 từng lớp** và **khoảng tin cậy**. Dùng **val để gỡ lỗi**, chỉ chấm **test** khi nộp kết quả.

## 4. Các thí nghiệm

| Mã | Setup | Bắt buộc |
|---|---|---|
| E0 | Zero-shot, không huấn luyện | ✔ |
| E1 | LoRA chỉ LLM; ViT và projector đóng băng | ✔ |
| E2 | E1 + mở đầy đủ projector (`visual.merger`, `visual.deepstack_merger_list`, ≈ 109M tham số) | ✔ |
| E3 | E2 + LoRA trên các khối attention/MLP của ViT | ✔ |
| E4 | Ablation: chọn **một** biến (rank / kích thước dữ liệu / thang toạ độ) | ✔ (chọn 1) |
| E5 | GRPO với reward IoU | tuỳ chọn, +10 điểm |

Mọi setup dùng **QLoRA 4-bit** (NF4). Riêng ViT và `lm_head` **không lượng tử hoá** (xem GUIDE bước 3 để biết vì sao).

## 5. Chuẩn bị (≈ 30 phút)

### Kaggle Notebook (Phạm Văn Kiên — 2A202602590)

1. Import `lab_day10.ipynb` vào Kaggle Notebook; chọn GPU **T4** (hoặc T4 x2) và bật **Internet** trong Settings.
2. Dùng **Add Input** để gắn dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`. Không cần nhập khóa API khi đọc dataset đã gắn.
3. Chạy các ô thiết lập: notebook clone repo đề vào `/kaggle/working/day10`, cài thư viện và chạy `prepare_data.py --source` trên dataset trong `/kaggle/input`. Model dùng GPU số 0; hai T4 không tự gộp VRAM.
4. Notebook đã nhúng mã trong `kaggle_vlm.py`, chạy lần lượt các ô để thực hiện E0–E3 và E4 (mặc định so sánh 10%/100% dữ liệu, chọn setup bằng F1 validation). Mỗi setup dùng batch 1, tích luỹ 8 lần để giữ effective batch 8 và giảm VRAM. Công cụ gốc trong `scripts/` và `tests/` được giữ nguyên.
5. Lưu adapter và projector trong `work/`, tải checkpoint sau từng thí nghiệm. Dùng **Save Version → Save & Run All**, rồi tải kết quả từ Output của phiên bản đã lưu. Ô cuối sinh `submission_2A202602590.zip` gồm số liệu thật, nhật ký tài nguyên và 6 ảnh có box dự đoán/nhãn gốc. Hoàn thiện 10 câu trong `REPORT.md` và tải notebook đã chạy riêng để giữ output; không commit trọng số hay dữ liệu.

Mã sinh viên: `kaggle_vlm.py`. Kiểm tra CPU: `python verify_student_code.py`. Sau khi sửa mã, chạy `python build_kaggle_notebook.py` để cập nhật bản nhúng trong notebook. Nếu dùng terminal Kaggle trong thư mục dự án đã chuẩn bị dữ liệu: `python kaggle_vlm.py --run-all --ablation data_size`. E5 GRPO không chạy mặc định; hàm reward IoU đã được viết và kiểm tra. Suy luận bị ngắt có thể tiếp tục; huấn luyện bị ngắt chạy lại từ đầu, còn setup đã hoàn tất được dùng lại khi cấu hình và dữ liệu khớp.

Khi gradient fp16 tràn số, mã mới giảm AMP scale và thử lại cùng batch (tối đa 9 lần), chỉ tăng bước và lịch LR sau cập nhật thành công; overflow được ghi vào log và `run.json`. Nếu phiên cũ dừng ở `Non-finite gradient`, chép mã `kaggle_amp_hotfix.py` vào một ô mới, chạy ô đó rồi chạy lại ô huấn luyện. Bản vá này giữ nguyên file mã và cache E0/E1; E2 bị ngắt vẫn phải chạy lại từ đầu vì checkpoint cũ không lưu optimizer/RNG.

Nếu Edit mở phiên mới và `/kaggle/working` trống, tạo notebook từ **Output phiên trước** hoặc gắn Output đó bằng **Add Input → Notebooks**. Dùng notebook mới đã sửa và vẫn gắn dataset mặt đường. Ô dữ liệu tự chép `work/` và `report/` về thư mục làm việc, kiểm tra mã model/nhãn/split rồi giữ fingerprint của phiên trước để dùng lại E0/E1. Cần Output có `kaggle_vlm.py` và `work/e1/final/adapter_model.safetensors`, không chỉ file log. E2 chạy lại với vòng AMP mới; không khôi phục optimizer của checkpoint E2 cũ.

Nếu E2 đã hoàn tất và lỗi xuất hiện ở E3, gắn Output mới nhất để dùng lại cả E2. Khi Output mới thiếu file nguồn khớp fingerprint E1, gắn thêm Output E1 gốc; mã sẽ chọn Output có nhiều thí nghiệm hoàn tất nhất và lưu nguồn checkpoint vào `report/provenance/` cho lần khôi phục tiếp theo. Môi trường Kaggle có `torchao==0.10.0` gây lỗi PEFT khi gắn LoRA lên ViT: bước thiết lập gỡ backend tùy chọn này (lab dùng bitsandbytes) và kiểm tra tạo adapter Linear trước khi chạy model.

Nếu chỉ gắn Output cũ và chưa gắn lại dataset gốc, ô dữ liệu có thể dùng `day10/data/images/` và `day10/data/splits/` trong Output. Mã kiểm tra ảnh, split và `dataset_version` trước khi chép về workspace. Nếu Output không có đủ dữ liệu, dùng Add Input → Datasets để gắn lại dataset mặt đường; file log hoặc checkpoint riêng không thay thế dữ liệu ảnh.

### Colab (hướng dẫn gốc)

1. Mở `lab_day10.ipynb` trên Colab: Runtime → *Change runtime type* → **T4 GPU**. Mount Google Drive để lưu adapter (session miễn phí có thể bị ngắt).
2. Tạo **Colab Secrets** tên `KAGGLE_USERNAME` và `KAGGLE_KEY` bằng tài khoản Kaggle của bạn. **Không ghi khoá vào notebook hay commit.**
3. `pip install -r requirements-colab.txt`, rồi `python scripts/prepare_data.py --kaggle`.
   Không có tài khoản Kaggle? Tải zip dataset về tay và dùng `--zip <đường dẫn>` hoặc `--source <thư mục đã giải nén>`.
4. `python -m pytest -q` để chắc công cụ của đề chạy đúng trên máy bạn.

## 6. Yêu cầu và cách nộp

Kết quả đã hoàn tất E0–E4 cho **Phạm Văn Kiên — 2A202602590**: E3 đạt F1 **0,4333**, CI95 **0,3894–0,4748**. Xem [báo cáo đã điền](REPORT.md), [notebook đã chạy có output](lab_day10_executed.ipynb) và [phân tích tái chấm](report/diagnostics/analysis.json). `lab_day10.ipynb` là bản để chạy mới; notebook executed là bằng chứng của phiên Kaggle đã kết thúc. Gói `submission_2A202602590_completed.zip` chứa báo cáo, mã, notebook executed, evidence, log và sáu ảnh; không chứa dữ liệu/trọng số. Tạo lại bằng `python package_submission.py`.

Kiểm tra mã bằng `python verify_student_code.py`. Test `test_report_template_has_student_markers_and_ten_questions` của đề yêu cầu báo cáo còn placeholder, nên không còn phù hợp sau khi điền; giữ nguyên test và ghi nhận riêng, không thêm placeholder để làm test xanh. Trên Windows có thể dùng thư mục tạm mới trong `work/` cho pytest nếu system Temp bị từ chối truy cập. Bộ kiểm tra bài nộp vẫn yêu cầu ít nhất năm commit có nội dung thực tế.

- Colab **T4 16 GB**. Mỗi lần huấn luyện ước tính **20–30 phút** (300 bước; con số ước lượng, hãy ghi thời gian thật của bạn vào báo cáo). Nếu thấy hết VRAM hoặc quá chậm, xem bảng "Khi gặp sự cố" trong GUIDE.
- Nộp **repo GitHub công khai** (≥ 5 commit) gồm: `report/evidence/e0–e3.json`, ít nhất một `e4*.json`, `REPORT.md` đã điền, ảnh minh hoạ, notebook/mã của bạn.
- Trước khi nộp: `python scripts/check_submission.py` không còn FAIL.
- **Không commit**: `data/`, `work/`, `.env`, trọng số (`*.safetensors`, `*.bin`, `*.pt`).
- **Không sửa** `scripts/` và `tests/` của đề.
- Làm cá nhân: bạn có thể thảo luận ý tưởng, nhưng mã, số liệu và báo cáo phải là của bạn. Giảng viên đối chiếu các repo với nhau và có thể gọi **vấn đáp 5 phút**.
