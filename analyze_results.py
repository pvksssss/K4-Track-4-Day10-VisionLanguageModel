"""Recover small result artifacts and independently verify the Kaggle scores."""
import collections
import hashlib
import json
import math
import random
import sys
import zipfile
from pathlib import Path, PurePosixPath

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _prf, evaluate, parse_output

ROOT = Path(__file__).resolve().parent
KEYS = ('e0', 'e1', 'e2', 'e3', 'e4_data_size_0p1', 'e4_data_size_1p0')


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def extract_small_artifacts():
    recovered = []
    with zipfile.ZipFile(ROOT / 'results.zip') as archive:
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if item.is_dir() or path.parts[:1] != ('day10',):
                continue
            rel = PurePosixPath(*path.parts[1:])
            if rel.is_absolute() or '..' in rel.parts or any(':' in p for p in rel.parts):
                raise ValueError(f'Unsafe archive path: {item.filename}')
            wanted = str(rel).startswith('report/') or (
                str(rel).startswith('data/splits/') and rel.suffix in ('.json', '.jsonl')) or (
                rel.parts[:1] == ('work',) and len(rel.parts) == 2 and
                rel.name.endswith(('_test.jsonl', '_val.jsonl')))
            if not wanted:
                continue
            if item.file_size > 20_000_000:
                raise ValueError(f'Unexpected large evidence: {item.filename}')
            target = ROOT.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(item)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f'Refusing to replace different artifact: {rel}')
            target.write_bytes(payload)
            recovered.append({'path': str(rel), 'bytes': len(payload),
                              'sha256': hashlib.sha256(payload).hexdigest()})
    write_json(ROOT / 'report/diagnostics/recovered_artifacts.json', recovered)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    extract_small_artifacts()
    entries = read_jsonl(ROOT / 'data/splits/test.jsonl')
    predictions, counts, checks = {}, {}, {}
    for key in KEYS:
        rows = read_jsonl(ROOT / f'work/{key}_test.jsonl')
        pred = {r['image_id']: r['raw'] for r in rows}
        assert len(pred) == len(rows) == len(entries)
        actual = evaluate(entries, pred)
        expected = json.loads((ROOT / f'report/evidence/{key}.json').read_text(encoding='utf-8'))
        def equal(a, b):
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, list):
                return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
            if isinstance(a, float):
                return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            return a == b
        for field in actual:
            assert equal(actual[field], expected[field]), (key, field, actual[field], expected[field])
        checks[key] = {'all_score_fields_match': True, 'f1': actual['micro']['f1']}
        predictions[key] = pred
        counts[key] = []
        for entry in entries:
            per, _ = _match(_gt_boxes(entry), parse_output(pred[entry['image_id']]) or [])
            counts[key].append([sum(v[k] for v in per.values()) for k in range(3)])
    correct = checks['e3']['f1']
    wrong = evaluate(entries, predictions['e3'], coord='pixel')
    pixel_predictions = {}
    for e in entries:
        parsed = parse_output(predictions['e3'][e['image_id']])
        pixel_predictions[e['image_id']] = json.dumps([
            {'label': p['label'], 'bbox_2d': [v*s/1000 for v, s in zip(
                p['bbox'], (e['width'], e['height'], e['width'], e['height']))]}
            for p in parsed])
    pixel_correct = evaluate(entries, pixel_predictions, coord='pixel')
    diagnostic = {'score_verification': checks,
        'coordinate_check': {'norm1000_correct_f1': correct,
            'norm1000_mistaken_as_pixel_f1': wrong['micro']['f1'],
            'norm1000_mistaken_as_pixel_counts': wrong['micro'],
            'converted_to_pixel_and_scored_as_pixel_f1': pixel_correct['micro']['f1'],
            'image_dimensions': dict(collections.Counter(f"{e['width']}x{e['height']}" for e in entries))},
        'e4_full_predictions_equal_e3': predictions['e3'] == predictions['e4_data_size_1p0']}
    paired = {}
    for a, b in [('e1', 'e2'), ('e2', 'e3'), ('e4_data_size_0p1', 'e3')]:
        rng, vals = random.Random(0), []
        for _ in range(1000):
            indices = [rng.randrange(len(entries)) for _ in entries]
            f1 = []
            for key in (a, b):
                total = [sum(counts[key][i][k] for i in indices) for k in range(3)]
                f1.append(_prf(*total)['f1'])
            vals.append(f1[1]-f1[0])
        vals.sort()
        paired[f'{b}_minus_{a}'] = {'delta': checks[b]['f1']-checks[a]['f1'],
            'ci95': [vals[25], vals[974]], 'method': 'paired image bootstrap; 1000; seed=0'}
    diagnostic['paired_f1_differences'] = paired
    split_info = {}
    for split in ('train', 'val', 'test'):
        rows = read_jsonl(ROOT / f'data/splits/{split}.jsonl')
        split_info[split] = {'images': len(rows), 'sessions': sorted({r['session'] for r in rows}),
            'boxes': dict(collections.Counter(o['label'] for r in rows for o in r['objects']))}
    diagnostic['splits'] = split_info
    diagnostic['sessions_disjoint'] = all(not (set(split_info[a]['sessions']) & set(split_info[b]['sessions']))
        for a, b in [('train','val'), ('train','test'), ('val','test')])
    size_stats = {}
    for label in ('pothole', 'crack', 'manhole'):
        boxes = [o for e in entries for o in e['objects'] if o['label'] == label]
        def median(values):
            values = sorted(values)
            n = len(values)
            return values[n//2] if n % 2 else (values[n//2-1]+values[n//2])/2
        size_stats[label] = {'count': len(boxes),
            'median_area_percent': median([o['yolo'][2]*o['yolo'][3]*100 for o in boxes]),
            'median_short_side_norm1000': median([min(o['yolo'][2:])*1000 for o in boxes]),
            'median_aspect_ratio': median([max(o['yolo'][2:])/min(o['yolo'][2:]) for o in boxes])}
    diagnostic['test_box_size'] = size_stats
    invalid = []
    figure_ids = ['vlcsnap-00202', 'vlcsnap-00204', 'vlcsnap-00206',
        'vlcsnap-2025-02-19-17h08m59s876', 'vlcsnap-2025-02-18-23h30m29s649',
        'vlcsnap-2025-02-19-17h03m29s526']
    figures = []
    for e in entries:
        raw = predictions['e0'][e['image_id']]
        if parse_output(raw) is None:
            invalid.append({'image_id': e['image_id'], 'length': len(raw),
                            'start': raw[:500], 'end': raw[-500:]})
        if e['image_id'] in figure_ids:
            per, _ = _match(_gt_boxes(e), parse_output(predictions['e3'][e['image_id']]))
            figures.append({'image_id': e['image_id'], 'per_class_counts': per,
                'gt': _gt_boxes(e), 'pred': parse_output(predictions['e3'][e['image_id']]),
                'e0_raw': raw})
    diagnostic['e0_invalid_examples'] = invalid
    diagnostic['figures'] = figures
    write_json(ROOT / 'report/diagnostics/analysis.json', diagnostic)
    print(json.dumps({k:v for k,v in diagnostic.items() if k not in ('figures','e0_invalid_examples')}, indent=2))
    print('First GT row:', json.dumps(entries[0], ensure_ascii=False))
    print('E0 invalid examples:', json.dumps(invalid[:3], ensure_ascii=False))
    for f in figures:
        print('FIGURE:', json.dumps(f, ensure_ascii=False))


if __name__ == '__main__':
    main()
