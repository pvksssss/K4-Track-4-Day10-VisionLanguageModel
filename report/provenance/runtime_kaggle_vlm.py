"""Student implementation of the Day 10 lab; official scripts stay unchanged.

Run on Kaggle: python kaggle_vlm.py --run-all
No GPU libraries are imported until a model or training dataset is used.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
CLASSES = ('pothole', 'crack', 'manhole')
STUDENT_NAME = 'Phạm Văn Kiên'
STUDENT_ID = '2A202602590'
# Transformers 5 matches skip patterns against the full module path.
QUANTIZATION_SKIP_MODULES = ['model.visual', 'visual', 'lm_head']


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def yolo_to_box(label, yolo, width=640, height=360, coord='norm1000'):
    """Round boxes; expand tiny positive boxes by one unit, omit zero-area labels."""
    if label not in CLASSES or coord not in ('norm1000', 'pixel'):
        raise ValueError('Unknown class or coordinate scale')
    xc, yc, w, h = yolo
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in yolo):
        raise ValueError('YOLO values must be finite and within [0, 1]')
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    if w == 0 or h == 0:
        return None
    sx, sy = (1000, 1000) if coord == 'norm1000' else (width, height)
    box = [round(max(0, min(1, v)) * scale) for v, scale in zip(
        (xc-w/2, yc-h/2, xc+w/2, yc+h/2), (sx, sy, sx, sy))]
    for lo, hi, limit in ((0, 2, sx), (1, 3, sy)):
        if box[hi] <= box[lo]:
            # Keep ordinary round() results unchanged. At the right/bottom edge,
            # shift the lower endpoint inward rather than exceeding the scale.
            box[lo] = min(box[lo], limit - 1)
            box[hi] = box[lo] + 1
    return {'label': label, 'bbox_2d': box}


def objects_to_answer(objects, width=640, height=360, coord='norm1000'):
    boxes = [yolo_to_box(o['label'], o['yolo'], width, height, coord) for o in objects]
    boxes = [b for b in boxes if b is not None]
    boxes.sort(key=lambda o: (o['bbox_2d'][1], o['bbox_2d'][0], o['label']))
    return json.dumps(boxes, separators=(',', ':'))


def audit_annotations(rows, coord='norm1000'):
    """Report label repairs before loading the GPU model; GT files stay unchanged."""
    stats = {'images': len(rows), 'boxes': 0, 'expanded_tiny_boxes': 0,
             'zero_area_labels_omitted_from_training_answers': 0, 'coord': coord}
    for row in rows:
        sx, sy = (1000, 1000) if coord == 'norm1000' else (row['width'], row['height'])
        for obj in row['objects']:
            stats['boxes'] += 1
            converted = yolo_to_box(obj['label'], obj['yolo'], row['width'], row['height'], coord)
            if converted is None:
                stats['zero_area_labels_omitted_from_training_answers'] += 1
                continue
            xc, yc, w, h = obj['yolo']
            rounded = [round(max(0, min(1, v))*s) for v, s in zip(
                (xc-w/2, yc-h/2, xc+w/2, yc+h/2), (sx, sy, sx, sy))]
            stats['expanded_tiny_boxes'] += int(rounded[2] <= rounded[0] or rounded[3] <= rounded[1])
    return stats


def make_prompt(coord='norm1000', width=640, height=360):
    if coord not in ('norm1000', 'pixel'):
        raise ValueError('Unknown coordinate scale')
    scale = ('integer coordinates on a 0-1000 scale, normalized independently '
             'by image width and height') if coord == 'norm1000' else (
             f'integer pixel coordinates in the original {width}x{height} image')
    return ('Locate all road damage objects in this image. Use only the labels '
            'pothole, crack, manhole. Return only a JSON list of objects with '
            'keys "label" and "bbox_2d": [x1,y1,x2,y2]. Use ' + scale +
            '. Sort boxes from top to bottom, then left to right. '
            'Return [] if no objects are present.')


def user_message(image, prompt):
    return {'role': 'user', 'content': [
        {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}


def config_for(setup, batch_size=1, grad_accum=8):
    import yaml
    cfg = yaml.safe_load(Path(f'configs/{setup}.yaml').read_text(encoding='utf-8'))
    cfg.update(coord='norm1000', train_fraction=1.0, inference_batch_size=1,
               max_new_tokens=256)
    if setup != 'e0':
        # Same effective batch (8 images) for E1-E4; lower T4 memory use.
        cfg.update(batch_size=batch_size, grad_accum=grad_accum)
    return cfg


def fingerprint(cfg):
    meta = json.loads(Path('data/splits/meta.json').read_text(encoding='utf-8'))
    payload = {'config': cfg, 'dataset_version': meta['dataset_version'],
               'prompt': make_prompt(cfg['coord']),
               'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(cfg):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU T4 in Kaggle Settings before running model cells')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=QUANTIZATION_SKIP_MODULES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.get('model_id', MODEL_ID),
        quantization_config=quant, dtype=torch.float16, device_map={'': 0},
        attn_implementation='sdpa')
    validate_quantization(model)
    processor = AutoProcessor.from_pretrained(cfg.get('model_id', MODEL_ID))
    processor.tokenizer.padding_side = 'left'
    return model, processor


def validate_quantization(model):
    """Fail before inference/training if the vision tower or output head is 4-bit."""
    protected = [(n, p) for n, p in model.named_parameters()
        if n.startswith(('model.visual.', 'visual.', 'lm_head.'))]
    visual = [(n, p) for n, p in protected if 'visual.' in n]
    if not visual:
        raise RuntimeError('Cannot locate vision tower; inspect full model module names')
    bad = [n for n, p in protected if not p.is_floating_point() or
           type(p).__name__ in ('Params4bit', 'Int8Params')]
    if bad:
        raise RuntimeError(f'Vision/head quantization exclusion failed: {bad[:5]}')
    quantized_llm = [n for n, p in model.named_parameters()
        if 'language_model.' in n and type(p).__name__ == 'Params4bit']
    if not quantized_llm:
        raise RuntimeError('LLM has no 4-bit weights; check quantization configuration')
    print('Quantization verified: vision/head floating point; '
          f'{len(quantized_llm)} LLM weights in 4-bit.', flush=True)


def apply_setup(model, cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()
    llm = r'.*language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)'
    vit = r'.*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))'
    pattern = f'(?:{llm}|{vit})' if cfg['lora']['targets'] == 'llm+vit' else llm
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    expected = 348 if cfg['lora']['targets'] == 'llm+vit' else 252
    if len(matched) != expected:
        raise RuntimeError(f'Expected {expected} LoRA modules, found {len(matched)}: {matched[:10]}')
    model = get_peft_model(model, LoraConfig(r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'], lora_dropout=cfg['lora']['dropout'],
        target_modules=pattern, bias='none', task_type='CAUSAL_LM'))
    # GradScaler cannot unscale fp16 gradients; LoRA optimizer weights stay fp32.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    if cfg['train_projector']:
        found = 0
        for n, p in model.named_parameters():
            if 'visual.merger.' in n or 'visual.deepstack_merger_list.' in n:
                if not p.is_floating_point():
                    raise RuntimeError('Projector is quantized; visual must remain unquantized')
                p.data = p.data.to(torch.float32)
                p.requires_grad_(True)
                found += p.numel()
        if found < 100_000_000:
            raise RuntimeError(f'Unexpected projector parameter count: {found}')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, matched


def encode_example(processor, entry, coord='norm1000'):
    import torch
    from PIL import Image
    with Image.open(Path('data') / entry['image_path']) as im:
        image = im.convert('RGB')
    user = user_message(image, make_prompt(coord, entry['width'], entry['height']))
    answer = objects_to_answer(entry['objects'], entry['width'], entry['height'], coord)
    prompt = processor.apply_chat_template([user], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt')
    full = processor.apply_chat_template([user, {'role': 'assistant', 'content': answer}],
        tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt')
    n = prompt['input_ids'].shape[1]
    if not torch.equal(full['input_ids'][0, :n], prompt['input_ids'][0]):
        raise RuntimeError('Chat prompt is not a prefix of the full conversation')
    full['labels'] = full['input_ids'].clone()
    full['labels'][:, :n] = -100
    if not (full['labels'] != -100).any():
        raise RuntimeError('No answer tokens remain for loss')
    return full


def collate(batch, pad_token_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    if any(set(b) != set(batch[0]) for b in batch):
        raise RuntimeError('Processor returned inconsistent batch keys')
    out = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if key in ('input_ids', 'attention_mask', 'mm_token_type_ids', 'labels', 'token_type_ids'):
            pad = -100 if key == 'labels' else pad_token_id if key == 'input_ids' else 0
            out[key] = pad_sequence([v.squeeze(0) for v in values], batch_first=True, padding_value=pad)
        elif key in ('pixel_values', 'image_grid_thw'):
            out[key] = torch.cat(values, dim=0)
        else:
            raise RuntimeError(f'Unexpected processor field {key}; inspect installed transformers')
    return out


def selected_train_rows(cfg):
    rows = read_jsonl('data/splits/train.jsonl')
    random.Random(cfg['seed']).shuffle(rows)
    return rows[:max(1, int(len(rows) * cfg['train_fraction']))]


def save_checkpoint(model, processor, directory, cfg, step, run=None):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    extra = {n: p.detach().cpu() for n, p in model.named_parameters()
             if p.requires_grad and ('visual.merger.' in n or 'visual.deepstack_merger_list.' in n)}
    if cfg['train_projector'] and not extra:
        raise RuntimeError('Projector checkpoint is empty')
    if extra:
        torch.save(extra, directory / 'projector.pt')
    write_json(directory / 'manifest.json', {'fingerprint': fingerprint(cfg), 'config': cfg, 'step': step})
    if run is not None:
        write_json(directory / 'run.json', run)


def train(cfg, key):
    import torch
    from torch.utils.data import DataLoader
    output = Path('work') / key
    final = output / 'final'
    signature = fingerprint(cfg)
    if (final / 'run.json').exists():
        run = json.loads((final / 'run.json').read_text(encoding='utf-8'))
        if run.get('fingerprint') != signature:
            raise RuntimeError(f'{final} belongs to another configuration; use a new run name')
        print('Reuse completed training:', key)
        return run
    seed_everything(cfg['seed'])
    rows = selected_train_rows(cfg)
    annotation_stats = audit_annotations(rows, cfg['coord'])
    write_json(output / 'annotation_audit.json', annotation_stats)
    print('Annotation audit:', annotation_stats, flush=True)
    model, processor = load_model(cfg)
    try:
        model, matched = apply_setup(model, cfg)
        processor.tokenizer.padding_side = 'right'
        class Dataset:
            def __len__(self): return len(rows)
            def __getitem__(self, i): return encode_example(processor, rows[i], cfg['coord'])
        loader = DataLoader(Dataset(), batch_size=cfg['batch_size'], shuffle=True,
            num_workers=0, generator=torch.Generator().manual_seed(cfg['seed']),
            collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=0)
        warmup = max(1, round(cfg['max_steps'] * .05))
        def lr_factor(step):
            if step < warmup: return (step + 1) / warmup
            progress = (step - warmup) / max(1, cfg['max_steps'] - warmup)
            return .5 * (1 + math.cos(math.pi * min(1, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        scaler = torch.amp.GradScaler('cuda')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        iterator = iter(loader)
        history = []
        write_json(output / 'config.json', cfg)
        for step in range(1, cfg['max_steps'] + 1):
            total_loss = 0.
            for _ in range(cfg['grad_accum']):
                try: batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {k: v.to('cuda:0') for k, v in batch.items()}
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at step {step}; lower LR and inspect labels')
                total_loss += loss.detach().float().item() / cfg['grad_accum']
                scaler.scale(loss / cfg['grad_accum']).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Non-finite gradient at step {step}; restart with lower LR')
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            history.append({'step': step, 'loss': total_loss, 'lr': scheduler.get_last_lr()[0]})
            if step == 1 or step % 10 == 0:
                print(f'{key} step={step}/{cfg["max_steps"]} loss={total_loss:.4f}', flush=True)
                write_json(output / 'loss.json', history)
            if step % 100 == 0 and step != cfg['max_steps']:
                # One rolling checkpoint limits disk usage. Interrupted training restarts;
                # optimizer/RNG are intentionally not presented as resumable state.
                save_checkpoint(model, processor, output / 'checkpoint', cfg, step)
        torch.cuda.synchronize()
        run = {'trainable_params': sum(p.numel() for p in params),
               'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3,
               'train_minutes': (time.perf_counter() - started) / 60,
               'fingerprint': signature, 'steps': cfg['max_steps'], 'train_images': len(rows),
               'matched_modules': len(matched), 'config': cfg,
               'annotation_audit': annotation_stats,
               'student_name': STUDENT_NAME, 'student_id': STUDENT_ID}
        save_checkpoint(model, processor, final, cfg, cfg['max_steps'], run)
        write_json(output / 'loss.json', history)
        print(json.dumps(run, ensure_ascii=False, indent=2))
        return run
    finally:
        # Release the model even after OOM; each experiment starts from the same base.
        if 'optimizer' in locals(): del optimizer
        if 'scheduler' in locals(): del scheduler
        if 'iterator' in locals(): del iterator
        if 'params' in locals(): del params
        if 'batch' in locals(): del batch
        if 'loss' in locals(): del loss
        if 'loader' in locals(): del loader
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def restore_model(cfg, key):
    import torch
    from peft import PeftModel
    model, processor = load_model(cfg)
    if key != 'e0':
        final = Path('work') / key / 'final'
        manifest = json.loads((final / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['fingerprint'] != fingerprint(cfg):
            raise RuntimeError('Adapter configuration or dataset version does not match')
        model = PeftModel.from_pretrained(model, final, is_trainable=False)
        if cfg['train_projector']:
            extra = torch.load(final / 'projector.pt', map_location='cpu', weights_only=True)
            expected = {n for n, _ in model.named_parameters()
                        if 'visual.merger.' in n or 'visual.deepstack_merger_list.' in n}
            if set(extra) != expected:
                raise RuntimeError('Projector checkpoint names differ from loaded model')
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in extra:
                        p.data = p.data.float()
                        p.copy_(extra[n].to(p.device))
    model.eval()
    model.config.use_cache = True
    return model, processor


def predict_and_score(cfg, key, split='test', prediction_key=None):
    import torch
    from PIL import Image
    pred_key = prediction_key or key
    pred_path = Path('work') / f'{pred_key}_{split}.jsonl'
    marker = pred_path.with_suffix('.manifest.json')
    signature = fingerprint(cfg)
    expected_marker = {'fingerprint': signature, 'split': split, 'model_run': key}
    rows = read_jsonl(f'data/splits/{split}.jsonl')
    if pred_path.exists():
        if not marker.exists() or json.loads(marker.read_text(encoding='utf-8')) != expected_marker:
            raise RuntimeError(f'Stale predictions: {pred_path}; use a different run name')
        existing = read_jsonl(pred_path)
        ids = [r['image_id'] for r in existing]
        if len(ids) != len(set(ids)) or not set(ids) <= {r['image_id'] for r in rows}:
            raise RuntimeError('Predictions contain duplicate or foreign image IDs')
        done = set(ids)
    else:
        done = set()
        write_json(marker, expected_marker)
    pending = [r for r in rows if r['image_id'] not in done]
    resource_path = pred_path.with_suffix('.resources.json')
    if pending:
        model, processor = restore_model(cfg, key)
        try:
            previous = json.loads(resource_path.read_text(encoding='utf-8')) if resource_path.exists() else {}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            bs = cfg['inference_batch_size']
            with pred_path.open('a', encoding='utf-8') as f, torch.inference_mode():
                for start in range(0, len(pending), bs):
                    batch = pending[start:start+bs]
                    conversations = []
                    for entry in batch:
                        with Image.open(Path('data') / entry['image_path']) as im:
                            image = im.convert('RGB')
                        conversations.append([user_message(image, make_prompt(cfg['coord'],
                            entry['width'], entry['height']))])
                    inputs = processor.apply_chat_template(conversations, tokenize=True,
                        add_generation_prompt=True, return_dict=True, return_tensors='pt',
                        processor_kwargs={'padding': True})
                    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = model.generate(**inputs, max_new_tokens=cfg['max_new_tokens'], do_sample=False)
                    raw = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    for entry, text in zip(batch, raw):
                        f.write(json.dumps({'image_id': entry['image_id'], 'raw': text}, ensure_ascii=False)+'\n')
                    f.flush()
                    write_json(resource_path, {
                        'inference_minutes': previous.get('inference_minutes', 0) +
                            (time.perf_counter() - started) / 60,
                        'inference_peak_vram_gb': max(previous.get('inference_peak_vram_gb', 0),
                            torch.cuda.max_memory_allocated() / 1024**3)})
                    if start % 20 == 0:
                        print(f'{pred_key} {split}: {len(done)+start+len(batch)}/{len(rows)}', flush=True)
        finally:
            if 'inputs' in locals(): del inputs
            if 'output' in locals(): del output
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
    evidence = Path('report/evidence') / f'{pred_key}.json' if split == 'test' else (
        Path('work') / f'{pred_key}_val_score.json')
    cmd = [sys.executable, 'scripts/score.py', '--pred', str(pred_path), '--split', split,
           '--coord', cfg['coord'], '--name', pred_key.upper(), '--out', str(evidence)]
    if key != 'e0':
        run = json.loads((Path('work') / key / 'final/run.json').read_text(encoding='utf-8'))
        for k in ('trainable_params', 'peak_vram_gb', 'train_minutes'):
            cmd.extend(['--meta', f'{k}={run[k]}'])
    else:
        cmd.extend(['--meta', 'trainable_params=0', '--meta', 'train_minutes=0'])
    if resource_path.exists():
        resources = json.loads(resource_path.read_text(encoding='utf-8'))
        for k, v in resources.items(): cmd.extend(['--meta', f'{k}={v}'])
        if key == 'e0':
            cmd.extend(['--meta', f'peak_vram_gb={resources["inference_peak_vram_gb"]}'])
    subprocess.run(cmd, check=True)
    return json.loads(evidence.read_text(encoding='utf-8'))


def run_ablation(configs, variable='data_size'):
    choices = {'data_size': [.1, 1.0], 'rank': [4, 64], 'coord_scale': ['norm1000', 'pixel']}
    if variable not in choices: raise ValueError('Choose data_size, rank or coord_scale')
    validation = {k: predict_and_score(c, k, 'val')['micro']['f1'] for k, c in configs.items()}
    best = max(validation, key=validation.get)
    write_json('work/ablation_selection.json', {'selection_split': 'val', 'f1': validation,
        'base_setup': best, 'variable': variable, 'values': choices[variable]})
    for value in choices[variable]:
        cfg = copy.deepcopy(configs[best])
        if variable == 'rank': cfg['lora'].update(r=value, alpha=2*value)
        elif variable == 'data_size': cfg['train_fraction'] = value
        else: cfg['coord'] = value
        tag = f'e4_{variable}_{str(value).replace(".", "p")}'
        if cfg == configs[best]:
            # The 100% or norm1000 reference can reuse the already-trained base.
            predict_and_score(cfg, best, prediction_key=tag)
        else:
            train(cfg, tag)
            predict_and_score(cfg, tag)
    return best


def draw_examples(split='test', prediction_key='e1', count=6, coord='norm1000'):
    from PIL import Image, ImageDraw
    from scripts.score import _gt_boxes, _match, parse_output
    rows = read_jsonl(f'data/splits/{split}.jsonl')
    preds = {r['image_id']: r['raw'] for r in read_jsonl(f'work/{prediction_key}_{split}.jsonl')}
    ranked = []
    for row in rows:
        parsed = parse_output(preds[row['image_id']])
        norm = parsed or []
        if coord == 'pixel':
            norm = [{'label': p['label'], 'bbox': tuple(v/s*1000 for v,s in zip(
                p['bbox'], (row['width'], row['height'], row['width'], row['height'])))} for p in norm]
        per, _ = _match(_gt_boxes(row), norm)
        error = sum(v[1]+v[2] for v in per.values())
        ranked.append((error, row, norm, parsed is not None))
    ranked.sort(key=lambda x: (x[0], x[1]['image_id']))
    half = count//2
    chosen = ranked[:half] + ranked[max(half, len(ranked)-(count-half)):]
    directory = Path('report/figures')
    directory.mkdir(parents=True, exist_ok=True)
    captions = ['# Ground truth (green) and predictions (red)', '']
    for i, (error, row, pred, valid) in enumerate(chosen):
        with Image.open(Path('data') / row['image_path']) as im: image = im.convert('RGB')
        draw = ImageDraw.Draw(image)
        for boxes, color in ((_gt_boxes(row), 'lime'), (pred, 'red')):
            for box in boxes:
                xy = [v*s/1000 for v,s in zip(box['bbox'],
                    (row['width'], row['height'], row['width'], row['height']))]
                draw.rectangle(xy, outline=color, width=2)
                draw.text((xy[0], max(0, xy[1]-12)), box['label'], fill=color)
        file = f'{prediction_key}_{i+1}.jpg'
        image.save(directory / file)
        captions.extend([f'![{row["image_id"]}]({file})',
            f'{row["image_id"]}: FP + FN = {error}; valid JSON = {valid}.', ''])
    (directory / 'README.md').write_text('\n'.join(captions), encoding='utf-8')


def export_results():
    evidence = sorted(Path('report/evidence').glob('e*.json'))
    head = '| Run | Trainable params | Peak VRAM GB | Minutes | F1 | CI95 | FP/image | Pothole F1 | Crack F1 | Manhole F1 |'
    lines = [head, '|---|---|---|---|---|---|---|---|---|---|']
    for p in evidence:
        d = json.loads(p.read_text(encoding='utf-8'))
        m = d['meta']
        cells = [p.stem, str(m.get('trainable_params', 0)),
                 f'{m["peak_vram_gb"]:.2f}' if 'peak_vram_gb' in m else '-',
                 f'{m["train_minutes"]:.2f}' if 'train_minutes' in m else '-',
                 f'{d["micro"]["f1"]:.4f}', f'{d["f1_ci95"][0]:.4f} - {d["f1_ci95"][1]:.4f}',
                 f'{d["fp_per_image"]:.3f}'] + [f'{d["per_class"][c]["f1"]:.4f}' for c in CLASSES]
        lines.append('| ' + ' | '.join(cells) + ' |')
    Path('report/results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    meta = json.loads(Path('data/splits/meta.json').read_text(encoding='utf-8'))
    report = Path('REPORT.md')
    text = report.read_text(encoding='utf-8').replace('Họ tên: <điền họ tên>',
        f'Họ tên: {STUDENT_NAME}\nMSV: {STUDENT_ID}\nMôi trường thực nghiệm: Kaggle T4')
    text = re.sub(r'`dataset_version`:.*', f'`dataset_version`: {meta["dataset_version"]}', text, count=1)
    # Replace only the results table; measured answers still require personal analysis.
    text = re.sub(r'\| Mã \| Setup \|.*?(?=\n## 10)', '\n'.join(lines)+'\n', text, flags=re.S)
    report.write_text(text, encoding='utf-8')
    write_json('report/data_summary.json', meta)
    # Preserve small reproducibility logs in the submission, without any weights.
    for p in Path('work').glob('*/final/run.json'):
        write_json(Path('report/runs') / (p.parent.parent.name + '_run.json'), json.loads(p.read_text(encoding='utf-8')))
    for p in Path('work').glob('*/loss.json'):
        write_json(Path('report/runs') / (p.parent.name + '_loss.json'), json.loads(p.read_text(encoding='utf-8')))
    for p in Path('work').glob('*_val_score.json'):
        write_json(Path('report/validation') / p.name, json.loads(p.read_text(encoding='utf-8')))
    if Path('work/ablation_selection.json').exists():
        write_json('report/ablation_selection.json', json.loads(Path('work/ablation_selection.json').read_text(encoding='utf-8')))
    paths = [p for p in Path('report').rglob('*') if p.is_file()]
    paths += [Path('REPORT.md'), Path('kaggle_vlm.py'), Path('lab_day10.ipynb')]
    paths += [Path('verify_student_code.py'), Path('build_kaggle_notebook.py')]
    paths += list(Path('configs').glob('*.yaml'))
    target = Path('submission_2A202602590.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.exists(): z.write(p, str(p))
    print('Download:', target.resolve())
    print('REPORT.md questions require analysis of your real outputs; no results are fabricated.')
    return target


def iou_reward(completion, gt_boxes):
    """Optional E5 reward, following the official specification (no GRPO run)."""
    from scripts.score import iou_xyxy, parse_output
    pred = parse_output(completion)
    if pred is None: return 0.0
    if not gt_boxes and not pred: return 1.0
    pairs = sorted(((iou_xyxy(g['bbox'], p['bbox']), i, j)
        for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
        if g['label'] == p['label']), reverse=True)
    used_g, used_p, total = set(), set(), 0.
    for iou, i, j in pairs:
        if iou < .5: break
        if i in used_g or j in used_p: continue
        used_g.add(i)
        used_p.add(j)
        total += iou
    return .2 + .8 * total / max(len(gt_boxes), len(pred), 1)


def run_all(variable='data_size'):
    predict_and_score(config_for('e0'), 'e0')
    configs = {key: config_for(key) for key in ('e1', 'e2', 'e3')}
    for key, cfg in configs.items():
        train(cfg, key)
        predict_and_score(cfg, key)
    best = run_ablation(configs, variable)
    draw_examples(prediction_key=best)
    export_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--ablation', choices=['data_size', 'rank', 'coord_scale'], default='data_size')
    args = parser.parse_args()
    if args.run_all: run_all(args.ablation)
    else: parser.print_help()
