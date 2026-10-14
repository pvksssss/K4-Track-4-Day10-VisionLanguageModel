"""Embed student code in the notebook so a single .ipynb upload works on Kaggle."""
import json
import inspect
from pathlib import Path


def main():
    path = Path('lab_day10.ipynb')
    nb = json.loads(path.read_text(encoding='utf-8'))
    import kaggle_vlm as lab
    hotfix_code = inspect.getsource(lab.amp_optimizer_step) + '\n\n' + inspect.getsource(lab.train)
    hotfix = ('# Paste into a cell in the existing Kaggle notebook, then rerun the training cell.\n'
              '# Keep the source file unchanged so completed E0/E1 cache signatures remain valid.\n'
              'import kaggle_vlm as lab\n\n'
              "code = r'''" + hotfix_code + "'''\n"
              "exec(code, lab.__dict__)\n"
              "print('AMP recovery patched in memory. Rerun the training cell; E2 restarts, E1 is reused.')\n")
    Path('kaggle_amp_hotfix.py').write_text(hotfix, encoding='utf-8')
    while len(nb['cells']) < 23:
        nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': []})
    def setcell(index, text):
        nb['cells'][index]['source'] = text.splitlines(True)
        if nb['cells'][index]['cell_type'] == 'code':
            nb['cells'][index]['outputs'] = []
            nb['cells'][index]['execution_count'] = None

    setcell(0, '# Lab Day 10: Qwen3-VL-4B road damage grounding\n'
        'Họ tên: **Phạm Văn Kiên** · MSV: **2A202602590**\n\n'
        'Kaggle T4: executable E0–E4 implementation. Run cells in order. '
        'Metrics are generated only from real model predictions. E5 GRPO is optional '
        'and is not executed; its IoU reward has CPU checks.\n')
    setup = """from pathlib import Path
import os
import subprocess
import sys

if Path('/kaggle/working').is_dir():
    project = Path('/kaggle/working/day10')
    if not (project / 'scripts/prepare_data.py').is_file():
        subprocess.run(['git', 'clone',
            'https://github.com/pvksssss/K4-Track-4-Day10-VisionLanguageModel.git',
            str(project)], check=True)
    os.chdir(project)
else:
    project = Path.cwd()
    if not (project / 'scripts/prepare_data.py').is_file():
        raise RuntimeError('Run inside the project folder or on Kaggle.')
print('Project:', project)
subprocess.run(['nvidia-smi', '-L'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
               'requirements-colab.txt', 'pytest', 'matplotlib'], check=True)
# This lab uses bitsandbytes, not torchao. Kaggle's optional torchao 0.10.0
# makes PEFT 0.21.2 fail when adding LoRA to ordinary fp16 vision Linear layers.
if Path('/kaggle/working').is_dir():
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
import importlib
importlib.invalidate_caches()
from peft.import_utils import is_torchao_available
is_torchao_available.cache_clear()
if Path('/kaggle/working').is_dir() and is_torchao_available():
    raise RuntimeError('Unexpected torchao backend is still active; restart the kernel')
# Exercise ordinary Linear dispatch now, before hours of E0/E1/E2 execution.
from peft import LoraConfig, get_peft_model
import torch
probe = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2, dtype=torch.float16)),
    LoraConfig(r=1, lora_alpha=2, target_modules=['0']))
assert any('lora_A' in name for name, _ in probe.named_parameters())
del probe
print('PEFT fp16 Linear adapter dispatch preflight passed.')
"""
    setcell(2, setup)
    module = Path('kaggle_vlm.py').read_text(encoding='utf-8')
    checks = Path('verify_student_code.py').read_text(encoding='utf-8')
    recovery = Path('restore_kaggle_output.py').read_text(encoding='utf-8')
    builder = Path(__file__).read_text(encoding='utf-8')
    setcell(3, "STUDENT_NAME = 'Phạm Văn Kiên'\nSTUDENT_ID = '2A202602590'\n"
        "print(f'{STUDENT_NAME} - {STUDENT_ID}')\n\n"
        "# Embedded implementation survives cloning the original assignment repo.\n"
        f"Path('kaggle_vlm.py').write_text({module!r}, encoding='utf-8')\n"
        f"Path('verify_student_code.py').write_text({checks!r}, encoding='utf-8')\n"
        f"Path('restore_kaggle_output.py').write_text({recovery!r}, encoding='utf-8')\n"
        f"Path('build_kaggle_notebook.py').write_text({builder!r}, encoding='utf-8')\n"
        "# Replace the cloned template with this implemented notebook for ZIP export.\n"
        "subprocess.run([sys.executable, 'build_kaggle_notebook.py'], check=True)\n"
        "sys.path.insert(0, str(project))\n"
        "import importlib\nimport kaggle_vlm as lab\nimportlib.reload(lab)\n"
        "report_path = project / 'REPORT.md'\n"
        "text = report_path.read_text(encoding='utf-8')\n"
        "text = text.replace('Họ tên: <điền họ tên>', "
        "f'Họ tên: {STUDENT_NAME}\\nMSV: {STUDENT_ID}\\nMôi trường thực nghiệm: Kaggle T4')\n"
        "report_path.write_text(text, encoding='utf-8')\n")
    setcell(1, '## 0. Environment (Kaggle T4)\n'
        'Select GPU T4 (or T4 x2), enable Internet, and attach '
        'lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes with Add Input. '
        'The model uses GPU 0; two T4 cards do not pool VRAM.\n\n'
        'To reuse completed E0/E1/E2 from the previous failed run, create a notebook from its '
        'Output or attach that Output as Input. If it lacks prepared data/images '
        'and data/splits, attach the original road dataset too. '
        'The data cell restores completed experiments after '
        'checking model/data compatibility. If it requests the original E1 '
        'source Output, attach that earlier Output as well. Unfinished runs '
        'start over with AMP recovery.\n')
    setcell(4, '## 1. Data: Kaggle Input and session splits\n'
        'Use the raw road dataset from Add Input, or reuse complete prepared data '
        '(images and train/val/test splits) from the previous notebook Output.\n')
    setcell(5, """from restore_kaggle_output import prepare_or_restore_data, restore_from_output
prepare_or_restore_data()
lab = restore_from_output(lab)
""")
    setcell(6, "from collections import Counter\nfrom PIL import Image, ImageDraw\n"
        "import matplotlib.pyplot as plt\n\n"
        "splits = {s: lab.read_jsonl(f'data/splits/{s}.jsonl') for s in ('train', 'val', 'test')}\n"
        "sessions = {s: {r['session'] for r in rows} for s, rows in splits.items()}\n"
        "assert not sessions['train'] & sessions['val']\n"
        "assert not sessions['train'] & sessions['test']\n"
        "assert not sessions['val'] & sessions['test']\n"
        "for split, rows in splits.items():\n"
        "    print(split, len(rows), Counter(o['label'] for r in rows for o in r['objects']))\n"
        "    audit = lab.audit_annotations(rows)\n"
        "    print('Annotation audit:', audit)\n"
        "    lab.write_json(f'report/annotations/{split}.json', audit)\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 7))\n"
        "for ax, row in zip(axes.flat, splits['train'][:6]):\n"
        "    with Image.open(Path('data') / row['image_path']) as im: image = im.convert('RGB')\n"
        "    draw = ImageDraw.Draw(image)\n"
        "    boxes = __import__('json').loads(lab.objects_to_answer(row['objects'], coord='pixel'))\n"
        "    for box in boxes:\n"
        "        draw.rectangle(box['bbox_2d'], outline='lime', width=2)\n"
        "        draw.text(tuple(box['bbox_2d'][:2]), box['label'], fill='lime')\n"
        "    ax.imshow(image); ax.set_title(row['image_id'], fontsize=8); ax.axis('off')\n"
        "plt.tight_layout(); plt.show()\n")
    setcell(8, "# Implemented in the embedded kaggle_vlm module.\n"
        "yolo_to_box = lab.yolo_to_box\nobjects_to_answer = lab.objects_to_answer\n"
        "print(objects_to_answer(splits['train'][0]['objects']))\n"
        "subprocess.run([sys.executable, 'verify_student_code.py'], check=True)\n"
        "subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)\n")
    setcell(10, "cfg0 = lab.config_for('e0')\n"
        "e0 = lab.predict_and_score(cfg0, 'e0')\n"
        "for row in lab.read_jsonl('work/e0_test.jsonl')[:5]: print(row)\n")
    setcell(11, "print('E0 valid JSON rate:', e0['parse_valid_rate'])\n"
        "print('E0 micro metrics:', e0['micro'], 'CI95:', e0['f1_ci95'])\n")
    setcell(13, "from transformers import AutoProcessor\n"
        "import torch\n"
        "processor = AutoProcessor.from_pretrained(lab.MODEL_ID)\n"
        "processor.tokenizer.padding_side = 'right'\n"
        "encoded = lab.encode_example(processor, splits['train'][0])\n"
        "answer_tokens = encoded['input_ids'][encoded['labels'] != -100]\n"
        "print('Supervised tokens:', processor.decode(answer_tokens, skip_special_tokens=False))\n"
        "image_token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')\n"
        "vision_tokens = int((encoded['input_ids'] == image_token).sum())\n"
        "print('Vision token count:', vision_tokens)\n"
        "assert vision_tokens == 220, 'Inspect image resolution and processor settings'\n"
        "two = lab.collate([encoded, lab.encode_example(processor, splits['train'][1])], "
        "processor.tokenizer.pad_token_id)\n"
        "assert torch.all(two['labels'][two['attention_mask'] == 0] == -100)\n"
        "print({k: tuple(v.shape) for k, v in two.items()})\n"
        "del two, encoded, answer_tokens, processor\n")
    setcell(15, "# Consistent effective batch = 8 for all experiments.\n"
        "configs = {key: lab.config_for(key, batch_size=1, grad_accum=8) "
        "for key in ('e1', 'e2', 'e3')}\n"
        "for key, cfg in configs.items(): print(key, cfg)\n"
        "# apply_setup validates 252 LLM / 348 LLM+ViT target modules during training.\n"
        "# AMP retries the same batch on overflow; only successful updates count as steps.\n")
    setcell(17, "runs = {}\n"
        "for key, cfg in configs.items():\n"
        "    runs[key] = lab.train(cfg, key)\n"
        "    # Score immediately so results survive interruptions between setups.\n"
        "    lab.predict_and_score(cfg, key)\n")
    setcell(19, "import json\n"
        "for key in ('e0', 'e1', 'e2', 'e3'):\n"
        "    result = json.loads(Path(f'report/evidence/{key}.json').read_text())\n"
        "    print(key, result['micro'], 'CI95:', result['f1_ci95'], 'resources:', result['meta'])\n")
    setcell(21, "# Choose exactly one variable. Default: 10% versus 100% training data.\n"
        "ABLATION_VARIABLE = 'data_size'  # alternatives: 'rank', 'coord_scale'\n"
        "best_setup = lab.run_ablation(configs, ABLATION_VARIABLE)\n"
        "print('Selected using validation F1:', best_setup)\n"
        "lab.draw_examples(prediction_key=best_setup, count=6)\n"
        "# Optional E5: IoU reward is implemented and tested; GRPO is not run here.\n")
    setcell(22, '## Save and download results\n'
        'The next cell exports real metrics, six annotated images and resource/loss logs '
        'into submission_2A202602590.zip. Complete the ten REPORT.md questions with '
        'your own observations. Save the executed Kaggle notebook and download it separately '
        'to preserve outputs. Checkpoints in work/ are excluded from the submission ZIP; '
        'download them separately if needed. Interrupted inference continues at the next image; '
        'interrupted training starts over, while completed training is reused.\n')
    export_cell = {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
        'source': ["archive = lab.export_results()\n",
                   "print(Path('report/results.md').read_text(encoding='utf-8'))\n",
                   "from IPython.display import FileLink, display\n",
                   "display(FileLink(str(archive)))\n"]}
    nb['cells'] = nb['cells'][:23] + [export_cell]
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print('Generated self-contained Kaggle notebook:', path)


if __name__ == '__main__':
    main()
