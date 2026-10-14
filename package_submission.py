"""Package the finished report and small evidence, without data or weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'submission_2A202602590_completed.zip'
FILES = ('README.md', 'GUIDE.md', 'RUBRIC.md', 'REPORT.md', '.gitignore',
         '.env.example', 'pytest.ini', 'requirements-colab.txt',
         'lab_day10.ipynb', 'lab_day10_executed.ipynb', 'kaggle_vlm.py',
         'build_kaggle_notebook.py', 'restore_kaggle_output.py',
         'kaggle_amp_hotfix.py', 'kaggle_box_hotfix.py',
         'verify_student_code.py', 'analyze_results.py', 'package_submission.py')


def main():
    paths = [ROOT / p for p in FILES]
    for folder in ('configs', 'scripts', 'tests', 'report'):
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths = sorted(set(paths))
    assert all(p.exists() for p in paths)
    manifest = {}
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        assert rel.split('/')[0] not in ('data', 'work', '.git')
        assert p.suffix not in ('.pt', '.bin', '.safetensors', '.ckpt', '.zip')
        assert p.name != '.env'
        manifest[rel] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'bytes': p.stat().st_size}
    # Open in exclusive-create mode: do not replace a user-supplied archive.
    with zipfile.ZipFile(TARGET, 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.writestr('submission_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(TARGET) as archive:
        assert archive.testzip() is None
        for rel, info in manifest.items():
            assert hashlib.sha256(archive.read(rel)).hexdigest() == info['sha256']
    print(f'{TARGET.name}: {len(paths)} files; {TARGET.stat().st_size / 1024**2:.2f} MiB; hashes verified')


if __name__ == '__main__':
    main()
