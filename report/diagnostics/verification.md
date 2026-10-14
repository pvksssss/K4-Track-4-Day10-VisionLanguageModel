# Kiểm tra bản báo cáo hoàn thiện

Ngày kiểm tra: 14/10/2026, workspace Windows của Phạm Văn Kiên.

| Kiểm tra | Kết quả |
|---|---|
| Tái chấm sáu predictions JSONL E0–E4 bằng `scripts.score.evaluate` | Mọi trường khớp evidence, dung sai số thực 1e-12 |
| Notebook executed | 13/13 ô code đã chạy, 0 error output |
| Báo cáo | Đủ 10 câu, 6 ảnh, không còn placeholder |
| `python verify_student_code.py` | Pass: tọa độ, nhãn, mask/collate, quantization, AMP overflow và reward |
| Pytest với thư mục tạm mới trong workspace | 67 passed, 1 skipped, 1 failed |
| `python scripts/check_submission.py` sau commit | **0 FAIL, 0 WARN**; git commits 5/5, bằng chứng, metadata, báo cáo và forbidden files đều PASS |
| Công cụ gốc | `git diff -- scripts tests` rỗng |

Lệnh pytest: `python -m pytest -q --basetemp=work/pytest_results_review_20261014 --tb=short`. Không dùng lại thư mục basetemp này nếu muốn giữ nội dung tạm; chọn một tên mới cho lần kiểm tra tiếp theo. Lần chạy đầu dùng system Temp bị lỗi quyền Windows; lần trên đã khắc phục bằng thư mục mới trong workspace.

Test duy nhất fail là `tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions`, vì test của mẫu đề yêu cầu báo cáo còn placeholder. Báo cáo đã điền nên không còn đáp ứng điều kiện mẫu; không sửa test và không thêm placeholder. Bộ kiểm tra bài nộp chính thức lại yêu cầu xóa placeholder và đã PASS mục báo cáo.

Repo có một commit trước công việc này. Theo yêu cầu push của người dùng, các thay đổi được chia thành bốn commit có nội dung riêng: mã train/inference; notebook và khôi phục Output; kết quả thực nghiệm; báo cáo và đóng gói. Sau commit, kiểm tra chính thức đạt 5/5 commit và 0 FAIL, 0 WARN. Đây là các commit đóng gói những phần đã hoàn thành, không phải lịch sử commit tại thời điểm từng thí nghiệm đang chạy.

Gói nộp được tạo bằng `package_submission.py`, loại dữ liệu, `work/`, trọng số, `.git` và ZIP gốc. Script kiểm tra CRC và SHA-256 từng tệp; danh sách nằm trong `submission_manifest.json` bên trong gói. Notebook executed được giữ nguyên nội dung từ `notebook495605e4e4.ipynb`; notebook chạy mới nằm ở `lab_day10.ipynb`.

ZIP hoàn thiện cục bộ được tạo trước bước commit/push; bản ghi kiểm tra trong ZIP phản ánh thời điểm đó. Tệp này trên GitHub cập nhật kết quả kiểm tra sau commit. Không đưa ZIP, dữ liệu hoặc trọng số lên GitHub.
