"""Reuse completed experiments from previous Kaggle Notebook Outputs attached as Input.

The previous source module keeps its original cache fingerprint. Its core model,
labels and data functions must match the current implementation. Only the AMP
training loop and result exporter are patched, without relabelling old evidence.
"""
import ast
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

CORE_FUNCTIONS = (
    'yolo_to_box', 'objects_to_answer', 'audit_annotations', 'make_prompt',
    'user_message', 'config_for', 'fingerprint', 'seed_everything', 'load_model',
    'validate_quantization', 'apply_setup', 'encode_example', 'collate',
    'selected_train_rows', 'save_checkpoint', 'restore_model', 'predict_and_score',
)


def validate_prepared_data(directory):
    """Check saved splits and all referenced images before restoring cached runs."""
    from scripts.splitting import dataset_version
    directory = Path(directory).resolve()
    meta = json.loads((directory / 'splits/meta.json').read_text(encoding='utf-8'))
    image_splits, seen_images, seen_sessions = [], set(), {}
    for split in ('train', 'val', 'test'):
        rows = [json.loads(line) for line in (directory / f'splits/{split}.jsonl')
                .read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows: raise RuntimeError(f'Empty saved split: {split}')
        if len(rows) != meta['counts'][split]['images']:
            raise RuntimeError(f'Saved split image count mismatch: {split}')
        for row in rows:
            if row['image_id'] in seen_images:
                raise RuntimeError('Duplicate image in saved splits')
            seen_images.add(row['image_id'])
            session = row['session']
            if session in seen_sessions and seen_sessions[session] != split:
                raise RuntimeError('Saved data has a session in multiple splits')
            seen_sessions[session] = split
            image = (directory / row['image_path']).resolve()
            if not image.is_relative_to(directory) or not image.is_file():
                raise RuntimeError(f'Saved Output is missing an image: {row["image_path"]}')
            image_splits.append((row['image_id'], split))
    if dataset_version(image_splits) != meta['dataset_version']:
        raise RuntimeError('Saved dataset_version does not match image/split assignments')
    return meta


def prepare_or_restore_data(input_dir='/kaggle/input', output_dir='data'):
    """Use the raw road dataset, or validated prepared data from prior Output."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if (output_dir / 'splits/meta.json').is_file():
        meta = validate_prepared_data(output_dir)
        print('Reuse prepared data:', output_dir.resolve(), meta['dataset_version'])
        return
    raw = sorted({p.parent for p in input_dir.rglob('labels-YOLO')
        if p.is_dir() and (p.parent / 'images').is_dir()})
    if len(raw) > 1:
        raise RuntimeError('Multiple raw road datasets found; attach one copy')
    if raw:
        print('Raw dataset:', raw[0])
        subprocess.run([sys.executable, 'scripts/prepare_data.py', '--source', str(raw[0]),
                        '--out', str(output_dir)], check=True)
        validate_prepared_data(output_dir)
        return
    prepared = sorted({p.parent.parent for p in input_dir.rglob('meta.json')
        if p.parent.name == 'splits' and (p.parent.parent / 'images').is_dir()})
    valid, failures = [], []
    for directory in prepared:
        try: valid.append((directory, validate_prepared_data(directory)))
        except (RuntimeError, KeyError, ValueError, FileNotFoundError) as error:
            failures.append(f'{directory}: {error}')
    if not valid:
        raise RuntimeError('No raw dataset or complete prepared data found. Use Add Input > Datasets '
            'to attach lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes. '
            + ('Incomplete Output: ' + '; '.join(failures) if failures else ''))
    # Multiple versions of the same Output can carry identical prepared splits.
    signatures = {tuple(hashlib_file(directory / f'splits/{name}') for name in
        ('meta.json', 'train.jsonl', 'val.jsonl', 'test.jsonl')) for directory, _ in valid}
    if len(signatures) > 1:
        raise RuntimeError('Attached Outputs have different prepared data; attach the correct dataset')
    directory, meta = valid[0]
    shutil.copytree(directory, output_dir, dirs_exist_ok=True)
    print('Restored prepared data from Output:', directory, meta['dataset_version'])


def hashlib_file(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compatible_source(previous_source, current_source):
    def functions(source):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    old, new = functions(previous_source), functions(current_source)
    changed = [name for name in CORE_FUNCTIONS if old.get(name) != new.get(name)]
    for constant in ('MODEL_ID', 'CLASSES', 'QUANTIZATION_SKIP_MODULES'):
        def value(source):
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == constant for t in node.targets):
                    return ast.literal_eval(node.value)
            return None
        if value(previous_source) != value(current_source): changed.append(constant)
    if changed:
        raise RuntimeError('Previous Output has different model/data settings: ' + ', '.join(changed))


def extract_embedded_sources(notebook_path):
    """Read literal module snapshots from code cells, without executing a notebook."""
    notebook = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code': continue
        source = cell.get('source', [])
        if isinstance(source, list): source = ''.join(source)
        try: tree = ast.parse(source)
        except SyntaxError: continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args: continue
            callee = node.func
            if not isinstance(callee, ast.Attribute) or callee.attr != 'write_text': continue
            receiver = callee.value
            if not isinstance(receiver, ast.Call) or not receiver.args: continue
            filename, contents = receiver.args[0], node.args[0]
            if not isinstance(filename, ast.Constant) or filename.value != 'kaggle_vlm.py': continue
            if isinstance(contents, ast.Constant) and isinstance(contents.value, str):
                yield contents.value


def restore_from_output(current, input_dir='/kaggle/input', output_root=None):
    input_dir = Path(input_dir)
    if not input_dir.exists(): return current
    candidates = sorted({p.parent for p in input_dir.rglob('kaggle_vlm.py')
        if (p.parent / 'work/e1/final/run.json').is_file()})
    if not candidates:
        print('No previous E1 Output attached; experiments run from scratch.')
        return current
    destination = Path(output_root).resolve() if output_root else Path.cwd().resolve()
    # Prefer the Output with the most completed experiments. Older attached Output
    # can provide the source file used before in-memory AMP patches were applied.
    def completed(root):
        return len(list((root / 'work').glob('*/final/run.json')))
    most = max(completed(p) for p in candidates)
    best = [p for p in candidates if completed(p) == most]
    if len(best) != 1:
        raise RuntimeError('Multiple Outputs have the same completed run count; attach one latest Output')
    previous_root = best[0]
    run = json.loads((previous_root / 'work/e1/final/run.json').read_text(encoding='utf-8'))
    config = current.config_for('e1')
    if run.get('config', config) != config:
        raise RuntimeError('Previous E1 configuration differs from the current experiment')
    current_source = Path(current.__file__).read_text(encoding='utf-8')
    provenance_name = 'report/provenance/runtime_kaggle_vlm.py'
    source_paths = [previous_root / provenance_name, previous_root / 'kaggle_vlm.py']
    source_paths += [p / provenance_name for p in candidates] + [p / 'kaggle_vlm.py' for p in candidates]
    # Some Outputs retain the exact old source only inside the executed notebook.
    # Extract literals safely; never rerun its setup, training or shell cells.
    extracted = destination / 'work/source_candidates'
    for notebook_path in input_dir.rglob('*.ipynb'):
        try:
            for source in extract_embedded_sources(notebook_path):
                try: compatible_source(source, current_source)
                except RuntimeError: continue
                import hashlib
                digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
                path = extracted / (digest + '.py')
                extracted.mkdir(parents=True, exist_ok=True)
                if not path.exists(): path.write_text(source, encoding='utf-8')
                source_paths.append(path)
        except (ValueError, SyntaxError, KeyError):
            continue
    previous = None
    for previous_path in dict.fromkeys(source_paths):
        if not previous_path.is_file(): continue
        try:
            compatible_source(previous_path.read_text(encoding='utf-8'), current_source)
        except RuntimeError:
            continue
        spec = importlib.util.spec_from_file_location('day10_previous_output', previous_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.fingerprint(config) == run['fingerprint']:
            previous = module
            break
    if previous is None:
        print('Notebook Output found:', previous_root)
        print('Expected E1 fingerprint:', run['fingerprint'])
        raise RuntimeError('Cannot find the source file matching completed E1. Attach the original '
                           'E1 Output as a second Input alongside the newest E2 Output.')
    # Verify every completed run before copying or reusing it.
    for run_path in (previous_root / 'work').glob('*/final/run.json'):
        saved = json.loads(run_path.read_text(encoding='utf-8'))
        if 'config' in saved and saved['fingerprint'] != previous.fingerprint(saved['config']):
            raise RuntimeError(f'Incompatible checkpoint provenance: {run_path}')
    # Preserve every original manifest, source fingerprint and evidence value.
    for folder in ('work', 'report'):
        src, dest = previous_root / folder, destination / folder
        if src.is_dir():
            for p in src.rglob('*'):
                if not p.is_file(): continue
                target = dest / p.relative_to(src)
                if target.exists(): continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
    provenance = destination / provenance_name
    provenance.parent.mkdir(parents=True, exist_ok=True)
    if provenance.exists() and provenance.read_bytes() != previous_path.read_bytes():
        raise RuntimeError('Working directory already has different checkpoint provenance')
    if not provenance.exists(): shutil.copy2(previous_path, provenance)
    # Source remains read-only in Input, keeping old signatures reproducible.
    for function in ('amp_optimizer_step', 'train', 'export_results'):
        exec(inspect.getsource(getattr(current, function)), previous.__dict__)
    cfg = previous.config_for('e1')
    run = json.loads((destination / 'work/e1/final/run.json').read_text(encoding='utf-8'))
    if run['fingerprint'] != previous.fingerprint(cfg):
        raise RuntimeError('Restored E1 config or dataset version differs; cannot reuse its checkpoint')
    adapter = destination / 'work/e1/final/adapter_model.safetensors'
    if not adapter.is_file():
        raise RuntimeError('Previous E1 Output is missing adapter_model.safetensors')
    print('Restored previous Output:', previous_root)
    reused = [p.parent.parent.name for p in (destination / 'work').glob('*/final/run.json')]
    print('Reuse completed training:', sorted(reused))
    print('Unfinished experiments restart with AMP recovery; original fingerprints are preserved.')
    return previous
