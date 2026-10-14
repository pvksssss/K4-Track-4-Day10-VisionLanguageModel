"""Paste this file into a Kaggle cell before retrying the failed training cell.

Patches the old module in memory, preserving its source-based E0 cache signature.
Does not change dataset ground truth, checkpoint files or prediction files.
"""
import json
import kaggle_vlm as lab

if not hasattr(lab, '_box_before_hotfix'):
    lab._box_before_hotfix = lab.yolo_to_box


def fixed_box(label, yolo, width=640, height=360, coord='norm1000'):
    try:
        return lab._box_before_hotfix(label, yolo, width, height, coord)
    except ValueError as error:
        if str(error) != 'Box collapses after rounding; inspect its annotation':
            raise
        xc, yc, w, h = yolo
        if w == 0 or h == 0:
            return None
        sx, sy = (1000, 1000) if coord == 'norm1000' else (width, height)
        box = [round(max(0, min(1, v))*s) for v, s in zip(
            (xc-w/2, yc-h/2, xc+w/2, yc+h/2), (sx, sy, sx, sy))]
        for lo, hi, limit in ((0, 2, sx), (1, 3, sy)):
            if box[hi] <= box[lo]:
                box[lo] = min(box[lo], limit-1)
                box[hi] = box[lo]+1
        return {'label': label, 'bbox_2d': box}


def fixed_answer(objects, width=640, height=360, coord='norm1000'):
    boxes = [fixed_box(o['label'], o['yolo'], width, height, coord) for o in objects]
    boxes = [b for b in boxes if b is not None]
    boxes.sort(key=lambda b: (b['bbox_2d'][1], b['bbox_2d'][0], b['label']))
    return json.dumps(boxes, separators=(',', ':'))


lab.yolo_to_box = fixed_box
lab.objects_to_answer = fixed_answer
print('Small-box fix applied. Rerun the failed cell; keep the current session and E0 files.')
