# Báo cáo Lab Day 10 — Vision Language Model

Họ tên: Phạm Văn Kiên

MSV: 2A202602590

Môi trường thực nghiệm: Kaggle Notebook, GPU NVIDIA T4

Model: `Qwen/Qwen3-VL-4B-Instruct`

`dataset_version`: d10-21e4070301

## Bảng tổng hợp

Số liệu lấy từ [bằng chứng chấm chính thức](report/evidence/), cùng 311 ảnh test, F1 micro tại IoU ≥ 0,5 và đúng lớp, tọa độ `norm1000`. CI95 bootstrap theo ảnh 1.000 lần, seed 0. Không thiếu dự đoán ở bất kỳ run nào. Đã chấm lại cả sáu JSONL bằng `scripts.score.evaluate`; mọi trường kết quả khớp bằng chứng, dung sai 1e-12 cho phép cộng số thực.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GiB) | Phút huấn luyện | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot | 0 | 3,416 | 0 | 0,0720 | 0,0476–0,0986 | 0,707 | 0,1000 | 0,0256 | 0,1511 |
| E1 | LoRA LLM | 33.030.144 | 5,085 | 52,12 | 0,2711 | 0,2332–0,3125 | 1,161 | 0,1949 | 0,1960 | 0,5197 |
| E2 | E1 + toàn bộ projector | 142.135.296 | 6,746 | 52,33 | 0,3324 | 0,2889–0,3758 | 1,029 | 0,2404 | 0,2446 | 0,6000 |
| E3 | E2 + LoRA ViT | 148.426.752 | 6,882 | 59,62 | 0,4333 | 0,3894–0,4748 | 0,846 | 0,3714 | 0,3474 | 0,6643 |
| E4–10% | E3, train_fraction=0,1 | 148.426.752 | 6,767 | 59,30 | 0,3108 | 0,2776–0,3484 | 1,534 | 0,2126 | 0,2303 | 0,5651 |
| E4–100% | E3, train_fraction=1,0; dùng lại checkpoint E3 | 148.426.752 | 6,882 | 59,62* | 0,4333 | 0,3894–0,4748 | 0,846 | 0,3714 | 0,3474 | 0,6643 |

VRAM là `torch.cuda.max_memory_allocated()/1024**3`, tức GiB cấp phát tensor, không phải tổng bộ nhớ GPU của `nvidia-smi`. E0 đo khi suy luận; các run có huấn luyện lấy VRAM huấn luyện. *E4–100% kế thừa chi phí train E3, không train thêm.* Thời gian suy luận test riêng: E0 29,10; E1 34,69; E2 34,40; E3 32,24; E4–10% 42,69; E4–100% 32,34 phút. Cột train không bao gồm chuẩn bị môi trường, validation và suy luận.

E1–E3 dùng rank 16, alpha 32, dropout 0,05, LR 2e-4, batch 1, gradient accumulation 8, 300 optimizer steps, seed 42; batch hiệu dụng 8. Suy luận batch 1, greedy, `max_new_tokens=256`. LLM NF4 4-bit, double quantization, compute fp16; ViT floating point, tham số cập nhật fp32. [Log train](report/runs/) lưu cấu hình và loss. E5 không chạy GRPO; hàm reward và kiểm tra CPU không phải bằng chứng hoàn thành E5.

## 10 câu hỏi

### Câu 1 — Kiến trúc và khối được cập nhật

ViT biến patch ảnh thành đặc trưng; projector/merger nối đặc trưng sang không gian LLM; LLM sinh JSON. Qwen3-VL có DeepStack đưa đặc trưng ảnh vào nhiều tầng, nên projector ở đây gồm `visual.merger` và `visual.deepstack_merger_list`.

Backbone khoảng **4,438 tỷ** tham số: LLM và embedding khoảng **4,02 tỷ**; phần thị giác khoảng **415 triệu**, gồm projector **109.105.152** và phần ViT còn lại khoảng **306 triệu**. Quy mô backbone là xấp xỉ kiến trúc; số tham số cập nhật dưới đây lấy trực tiếp từ run. Không nhầm số tham số adapter với backbone, hay số phần tử đóng gói 4-bit với số tham số logic.

| Setup | LLM | Projector | ViT | Tổng cập nhật |
|---|---|---|---|---:|
| E1 | LoRA 33.030.144; 36×7=252 module | Đóng băng | Đóng băng | 33.030.144 |
| E2 | Như E1 | Toàn bộ 109.105.152 | Các block đóng băng | 142.135.296 |
| E3 | Như E1 | Như E2 | LoRA 6.291.456; 24×4=96 module | 148.426.752 |

E3 khớp 348 module LoRA. Trọng số backbone LLM/ViT vẫn đóng băng. Notebook ghi `trainable params: 148,426,752 || all params: 4,477,137,408`; tổng sau gắn adapter bao gồm 39.321.600 tham số LoRA mới.

### Câu 2 — E0 và lỗi của model gốc

`parse_valid_rate` E0 = **222/311 = 0,7138**, tức 89 output không hợp lệ. E1 đạt 310/311 = 0,9968, E3 đạt 311/311. Hai nhóm lỗi nổi bật khi đối chiếu output và nhãn:

1. **JSON thiếu dấu đóng hoặc bị cắt khi sinh lặp box.** Toàn bộ output thô của `vlcsnap-00206` dưới đây thiếu `]`. Bộ chấm chấp nhận code fence, nên fence không gây lỗi:

````text
```json
[
    {"label": "manhole", "bbox_2d": [310, 670, 399, 714]
}
```
````

   `vlcsnap-00205` lặp nhiều crack; cuối output thật dừng tại `{"label": "crack", "bbox_2d": [359, 540, `, chưa hoàn thành JSON. Sinh lặp có thể chạm giới hạn 256 token và khiến cả ảnh bị parse thất bại. Không gán mọi invalid cho giới hạn token: ví dụ ngắn ở trên là lỗi cấu trúc.

2. **Box hợp lệ nhưng lặp, định vị không khớp hoặc bỏ sót.** E0 có 30 TP, 220 FP, 553 FN trên 583 GT; riêng crack có 6 TP, 163 FP, 293 FN. Output `vlcsnap-00202`:

```json
[{"label":"manhole","bbox_2d":[309,686,399,714]},
 {"label":"manhole","bbox_2d":[309,686,399,714]}]
```

GT xấp xỉ `[275,672,405,722]`; box E0 có IoU khoảng 0,389 < 0,5, nên hai dự đoán đều FP, GT là FN. Recall E0 chỉ 0,0515. Đúng định dạng chưa đảm bảo phát hiện đúng.

### Câu 3 — Kiểm chứng thang tọa độ 0–1000

Từ YOLO `[xc,yc,w,h]`, đổi sang góc: `[(xc-w/2)*1000,(yc-h/2)*1000,(xc+w/2)*1000,(yc+h/2)*1000]`. Với ảnh 640×360, đổi ngược x nhân 640/1000, y nhân 360/1000. Thang chuẩn hóa mô tả vị trí độc lập độ phân giải.

Tôi giữ nguyên dự đoán E3 và chạy ba phép chấm bằng bộ chấm gốc:

| Đầu ra và cách chấm | F1 | TP/FP/FN |
|---|---:|---|
| norm1000, chấm norm1000 | 0,4333 | 234/263/349 |
| norm1000, khai báo nhầm pixel | **0,0000** | 0/497/583 |
| Quy đổi sang pixel bằng số thực, chấm pixel | 0,4333 | 234/263/349 |

Tất cả 311 ảnh test đều 640×360. Chấm nhầm sẽ nhân lại x với 1000/640, y với 1000/360: vị trí và kích thước sai theo hai hệ số khác nhau. Chiều ngược lại cũng làm box co sai. Round-trip bằng số thực giữ kết quả; làm tròn pixel có thể ảnh hưởng box sát ngưỡng IoU. Số kiểm chứng nằm trong [analysis.json](report/diagnostics/analysis.json); tái chạy `python analyze_results.py` khi có ZIP Output gốc.

### Câu 4 — Mask loss

SFT học JSON với điều kiện đã có ảnh/prompt. Tôi encode hội thoại, kiểm tra prompt là tiền tố chính xác, đặt label prompt và padding bằng `-100`, giữ label JSON và `<|im_end|>`. Cross-entropy bỏ qua `-100`; gradient của đáp án vẫn truyền qua đường xử lý ảnh đến projector/LoRA ViT.

Notebook kiểm tra **220 token ảnh** cho ảnh 640×360. Giải mã vị trí `labels != -100` đúng JSON đáp án cộng `<|im_end|>`. Collate hai ví dụ: `input_ids`, `attention_mask`, `mm_token_type_ids`, `labels` shape `(2,574)`; `pixel_values` `(1760,1536)`, `image_grid_thw` `(2,3)`, nối theo chiều 0. Không pad patch ảnh như token văn bản; label padding là `-100`.

Nếu tính loss cả prompt, model phải học dự đoán câu hướng dẫn và placeholder ảnh; nhiều vị trí không phải đáp án có thể chi phối loss, giảm trọng số tương đối của học nhãn/box. Loss trên placeholder không phải tái tạo pixel. Tôi chưa chạy ablation bỏ mask nên không báo mức giảm F1 giả định. Kiểm tra supervised tokens xác nhận phạm vi loss, không đo lợi ích nhân quả của mask.

### Câu 5 — 4-bit và projector float32

Trọng số 4-bit bitsandbytes đóng gói kèm scale, dùng làm backbone đóng băng trong QLoRA. Chỉ bật `requires_grad=True` không biến chúng thành ma trận floating point để full fine-tuning bằng Adam. Có phương pháp dequantize/quantization-aware training khác, nhưng pipeline này không dùng.

Tôi loại vision tower và lm_head khỏi lượng tử hóa bằng `['model.visual','visual','lm_head']`, kiểm tra tên module thật. Log xác nhận vision/head floating point, **252 trọng số LLM 4-bit**. Tên `visual` từng không khớp `model.visual` ở bản thư viện đã chạy; kiểm tra runtime bắt lỗi trước train. E2/E3 đưa merger/DeepStack merger sang fp32 và mở gradient; E3 thêm LoRA ViT, các block gốc vẫn đóng băng.

Tham số cập nhật projector/LoRA giữ fp32 để tích lũy gradient, cập nhật nhỏ ổn định hơn và phù hợp AMP/GradScaler đang dùng: unscale gradient fp16 có thể lỗi. Đây là lựa chọn pipeline, không phải mọi hệ thống bắt buộc projector fp32. Forward dùng autocast fp16 trên T4; backbone thị giác đóng băng giữ fp16. Projector được lưu riêng và nạp lại cùng adapter khi đánh giá.

E2/E3 không overflow trong log cuối. E4–10% có một overflow step 137, giảm scale 4096 xuống 2048, tính lại **cùng batch**, không cập nhật trọng số/scheduler khi overflow. E1 được khôi phục từ run trước sửa AMP, nên chi tiết triển khai E1 và E2/E3 không hoàn toàn đồng nhất; log/provenance được giữ để minh bạch.

### Câu 6 — Split theo phiên quay

Frame cùng phiên có cùng đường, vật thể, góc camera, ánh sáng, có thể gần như bản sao. Random split từng ảnh có thể đưa frame gần nhau vào train/test: model ghi nhớ cảnh thay vì khái quát sang phiên mới, khiến F1 có vẻ cao hơn. Chưa chạy random split nên không báo số tăng giả định.

| Split | Số phiên | Số ảnh | Pothole | Crack | Manhole | Tổng box |
|---|---:|---:|---:|---:|---:|---:|
| Train | 13 | 1.422 | 1.074 | 1.750 | 691 | 3.515 |
| Val | 2 | 276 | 66 | 470 | 103 | 639 |
| Test | 3 | 311 | 121 | 299 | 163 | 583 |
| Tổng | 18 | 2.009 | 1.261 | 2.519 | 957 | 4.737 |

Đã kiểm tra giao tập session giữa ba cặp split đều rỗng. Test gồm `num-4`, `vlc-2025-02-18-23`, `vlc-2025-02-19-17`; danh sách đầy đủ trong diagnostics. Split dùng seed `day10`, salt 0, cùng version cho mọi run.

Audit có **một nhãn train diện tích 0**, bỏ khỏi đáp án huấn luyện vì không tạo được box hợp lệ. GT/split gốc giữ nguyên, tổng box train trên bảng vẫn 3.515. Val/test không có nhãn diện tích 0; không box nào phải mở rộng do làm tròn norm1000. Audit lưu trong `report/annotations/` và log run.

### Câu 7 — Lợi ích và chi phí E1/E2/E3

E1 tăng F1 so với E0 **0,1990**, vượt +0,10 của rubric; parse tăng 0,7138 lên 0,9968. E2 thêm 109.105.152 tham số projector, F1 +**0,0613**, VRAM +**1,660 GiB**, train +**0,21 phút**. E3 thêm 6.291.456 tham số LoRA ViT so với E2, F1 +**0,1009**, VRAM +**0,137 GiB**, train +**7,28 phút**. E3 giảm FP/ảnh 1,029 xuống 0,846, cải thiện cả ba lớp; có lợi trong ngân sách T4 lần chạy này.

CI biên E1/E2 chồng lấn không tự chứng minh chênh lệch bằng 0. Tôi bổ sung bootstrap ghép cặp trên **cùng chỉ số ảnh**, 1.000 lần, seed 0:

| Chênh lệch F1 | Ước lượng | CI95 ghép cặp theo ảnh |
|---|---:|---|
| E2 − E1 | +0,0613 | +0,0221 đến +0,0985 |
| E3 − E2 | +0,1009 | +0,0584 đến +0,1465 |

Các khoảng hỗ trợ chiều cải thiện trên test hiện tại, nhưng chưa xử lý tương quan frame cùng phiên; mỗi cấu hình chỉ một seed. Không bảo đảm cho mọi tuyến đường/lần train. VRAM thấp nhờ adapter, lượng tử hóa LLM và gradient checkpointing.

### Câu 8 — Độ bất định trên 311 ảnh

CI95 E3 **[0,3894; 0,4748]**, rộng **0,0854**, khoảng ±0,043 quanh 0,4333. Độ rộng E0/E1/E2 lần lượt **0,0511 / 0,0793 / 0,0869**; E4–10% **0,0708**. Chênh lệch vài phần trăm F1 cần phân phối chênh lệch ghép cặp, không chỉ hai điểm ước lượng hay độ chồng lấn CI biên.

Micro F1 gộp TP/FP/FN, không phải trung bình F1 từng ảnh. Bộ chấm lấy lại 311 ảnh có hoàn lại mỗi lần rồi tính F1 gộp. CI mô tả bất định do lấy mẫu ảnh với model hiện có, không bao gồm seed train hay miền dữ liệu mới.

Chỉ có **ba phiên test**, frame trong phiên phụ thuộc nhau; coi 311 ảnh độc lập có thể cho CI quá hẹp. Chưa có CI theo cụm phiên đáng tin cậy hoặc nhiều lần train để kết luận rộng hơn. Cần bổ sung phiên độc lập và nhiều seed trước khi cam kết chất lượng triển khai.

### Câu 9 — Lỗi theo lớp và ảnh minh họa

E3 kém nhất ở **crack**, F1 0,3474; pothole 0,3714; manhole 0,6643. Crack: 103 TP, **191 FP**, 196 FN; pothole: 39 TP, 50 FP, 82 FN; manhole: 92 TP, 22 FP, 71 FN. Crack chiếm khoảng 73% FP, dù có nhiều mẫu train nhất (1.750 box); không thể chỉ giải thích bằng thiếu mẫu.

GT test có median diện tích box pothole **0,419%**, crack **1,981%**, manhole **0,669%** ảnh; median cạnh ngắn norm1000 **55,6 / 106,3 / 66,7**. Median tỉ lệ cạnh dài/ngắn crack khoảng **1,60**. Crack không phải lớp có median box nhỏ nhất. Vết nứt không đều, nối nhiều vùng, dễ nhầm mép vá đường; cách gom/tách box khác GT có thể làm IoU thấp. Pothole nhỏ, nằm trong vùng crack cũng gây nhầm lớp. Đây là giải thích từ ảnh/box đã quan sát, chưa phải ablation xác định nguyên nhân.

Sáu ảnh chọn ba ít lỗi nhất và ba nhiều lỗi nhất theo FP+FN, **không phải mẫu ngẫu nhiên** đại diện test. **Xanh lá: GT; đỏ: E3**. Match đúng lớp, IoU ≥ 0,5. Cả sáu đều JSON hợp lệ.

![E3 manhole đúng ở vlcsnap-00202](report/figures/e3_1.jpg)

1. `vlcsnap-00202`: manhole 1 TP, 0 FP, 0 FN. E3 khớp GT; E0 xuất hai box nhỏ trùng nhau không đạt IoU.

![E3 manhole đúng ở vlcsnap-00204](report/figures/e3_2.jpg)

2. `vlcsnap-00204`: manhole 1 TP, 0 FP, 0 FN; box đúng vùng nắp cống.

![E3 manhole đúng ở vlcsnap-00206](report/figures/e3_3.jpg)

3. `vlcsnap-00206`: manhole 1 TP, 0 FP, 0 FN; E0 thiếu dấu `]` ở câu 2, E3 sửa cả định dạng/định vị.

![E3 bỏ sót pothole và định vị crack sai](report/figures/e3_4.jpg)

4. `vlcsnap-2025-02-19-17h08m59s876`: GT có 3 pothole, 2 crack; model chỉ xuất 2 crack, đều không match. Tổng 0 TP, 2 FP, 5 FN. Một box trùm vùng pothole nhưng nhãn crack; một box ở mép đường trái không khớp GT.

![E3 nhiều dự đoán thừa trong cảnh ngược sáng](report/figures/e3_5.jpg)

5. `vlcsnap-2025-02-18-23h30m29s649`: GT 3 crack; crack 1 TP, 3 FP, 2 FN, thêm 1 pothole FP và 2 manhole FP. Tổng 1 TP, 6 FP, 2 FN. Cảnh chói, mặt đường nhiều texture; model gán pothole và manhole cho gần cùng vùng phía trên. Ảnh cho thấy nhầm lớp/dự đoán thừa, không chứng minh ánh sáng là nguyên nhân duy nhất.

![E3 tách nhỏ vùng và nhầm pothole thành crack](report/figures/e3_6.jpg)

6. `vlcsnap-2025-02-19-17h03m29s526`: GT 3 pothole, 1 crack lớn; model xuất 7 crack nhỏ, không match đúng lớp/IoU. Tổng 0 TP, 7 FP, 4 FN. Box đỏ bám nhiều vùng pothole nhưng nhãn crack, đồng thời tách nhỏ crack GT lớn.

### Câu 10 — E4 chỉ đổi lượng dữ liệu

Chọn nền bằng **validation**, không bằng test: F1 val E1 0,2153, E2 0,2719, E3 **0,3644**, chọn E3. Chỉ đổi `train_fraction` **0,1 / 1,0**: 142 ảnh/339 box so với 1.422 ảnh/3.515 box. Rank, LR, batch hiệu dụng, 300 steps, seed, prompt, tọa độ, split giữ nguyên. Tập 10% được lặp nhiều hơn; đây là cùng ngân sách optimizer steps, không phải cùng số epoch.

E4–10% train từ backbone gốc với setup E3; bản 100% dùng lại **checkpoint E3 đã train**, rồi suy luận test riêng. Đã kiểm tra ánh xạ `image_id → raw` E4–100% giống hoàn toàn E3. Đây không phải lần train độc lập thứ hai để đo độ ổn định.

10% đạt F1 **0,3108**, CI95 **[0,2776; 0,3484]**, FP/ảnh **1,534**; toàn bộ đạt **0,4333**, CI95 **[0,3894; 0,4748]**, FP/ảnh **0,846**. Chênh lệch +**0,1226**; CI95 ghép cặp theo ảnh **[+0,0824; +0,1629]**. Loss cuối 10% **0,0259**, thấp hơn E3 **0,4564**, nhưng test kém hơn: phù hợp việc tập nhỏ dễ khớp và thiếu đa dạng, chưa chứng minh overfitting bằng đường validation loss đầy đủ.

Trong lần chạy này toàn bộ dữ liệu tốt hơn rõ rệt. Giới hạn: một seed, ba phiên test, frame phụ thuộc, chưa thử nhiều tập con 10%. Không kết luận thêm dữ liệu bất kỳ luôn giúp; cần dữ liệu mới đa dạng, nhãn nhất quán.

## Nhìn lại

Lỗi triển khai trước run cuối: box diện tích 0, skip lượng tử hóa không khớp tên, fp16 gradient/AMP overflow, PEFT kiểm tra torchao không tương thích dù dùng bitsandbytes. Mã kiểm tra lượng tử hóa sớm, bỏ nhãn train không hợp lệ có audit, giữ tham số cập nhật fp32, retry cùng batch khi overflow, gỡ torchao không dùng và chạy LoRA preflight. E1/E2 khôi phục từ Output với fingerprint/provenance; kernel mới mất biến RAM không bắt buộc train lại.

Bộ chấm không đảm bảo GT đầy đủ hoặc model đủ tốt để sử dụng thực tế. Vùng model thấy nhưng chưa gán nhãn vẫn tính FP; cần rà soát, không tự sửa GT để tăng điểm. `meanIoU` E3 **0,6959** chỉ tính 234 cặp match, không mô tả 349 box bỏ sót. JSON 100% hợp lệ và F1 0,4333 còn xa bộ phát hiện đáng tin cậy. Cần rà nhãn crack/pothole, thống nhất gom box, tăng phiên độc lập, chạy nhiều seed.

## Tệp tái kiểm tra

- [Notebook đã chạy có output](lab_day10_executed.ipynb); [notebook chạy mới](lab_day10.ipynb).
- [Mã train/inference](kaggle_vlm.py), [khôi phục Output](restore_kaggle_output.py), [kiểm tra mã](verify_student_code.py).
- [Bằng chứng](report/evidence/), [log train](report/runs/), [chọn E4](report/ablation_selection.json), [phân tích bổ sung](report/diagnostics/analysis.json).
- [Kết quả kiểm tra bài hoàn thiện](report/diagnostics/verification.md).
- `python scripts/check_submission.py` kiểm tra cấu trúc bài. Gói nộp nhẹ loại dữ liệu, `work/`, trọng số, `.git`, ZIP gốc. Tái chấm từ predictions cần ZIP Output gốc hoặc chạy notebook; metadata split giữ trong `report/data_summary.json`.
