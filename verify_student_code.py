"""CPU checks for student labels, coordinate conventions and optional reward."""
import json
import math
import re
import tempfile
from pathlib import Path
from kaggle_vlm import objects_to_answer, yolo_to_box, make_prompt, iou_reward
from scripts.score import evaluate, _gt_boxes


def main():
    entry = {'image_id': 'sample', 'width': 640, 'height': 360,
             'objects': [{'label': 'pothole', 'yolo': [.5, .5, .25, .2]}]}
    norm = objects_to_answer(entry['objects'])
    pixel = objects_to_answer(entry['objects'], coord='pixel')
    assert json.loads(norm)[0]['bbox_2d'] == [375, 400, 625, 600]
    assert json.loads(pixel)[0]['bbox_2d'] == [240, 144, 400, 216]
    assert objects_to_answer([]) == '[]'
    assert yolo_to_box('crack', [0, 0, .2, .2])['bbox_2d'] == [0, 0, 100, 100]
    for coord in ('norm1000', 'pixel'):
        for xc, yc in ((.5, .5), (0, 0), (1, 1), (.00001, .99999)):
            box = yolo_to_box('crack', [xc, yc, .000001, .000001], coord=coord)['bbox_2d']
            sx, sy = (1000, 1000) if coord == 'norm1000' else (640, 360)
            assert 0 <= box[0] < box[2] <= sx
            assert 0 <= box[1] < box[3] <= sy
        for w, h in ((0, .1), (.1, 0), (0, 0)):
            assert yolo_to_box('crack', [.5, .5, w, h], coord=coord) is None
    assert objects_to_answer([{'label': 'crack', 'yolo': [.5, .5, 0, .1]}]) == '[]'
    from kaggle_vlm import audit_annotations
    audit = audit_annotations([{'width':640, 'height':360, 'objects':[
        {'label': 'crack', 'yolo': [.5, .5, .000001, .000001]},
        {'label': 'pothole', 'yolo': [.5, .5, 0, .1]}]}])
    assert audit['expanded_tiny_boxes'] == 1
    assert audit['zero_area_labels_omitted_from_training_answers'] == 1
    for raw, scale in ((norm, 'norm1000'), (pixel, 'pixel')):
        score = evaluate([entry], {'sample': raw}, coord=scale, n_boot=100)
        assert score['micro']['f1'] == 1
    assert evaluate([entry], {'sample': pixel}, n_boot=100)['micro']['f1'] == 0
    objects = entry['objects'] + [{'label': 'manhole', 'yolo': [.2, .2, .1, .1]}]
    assert json.loads(objects_to_answer(objects))[0]['label'] == 'manhole'
    assert '0-1000' in make_prompt() and 'pixel' in make_prompt('pixel')
    gt = _gt_boxes(entry)
    assert math.isclose(iou_reward(norm, gt), 1)
    assert iou_reward('invalid', gt) == 0
    assert iou_reward('[]', []) == 1
    assert iou_reward('[]', gt) == .2
    wrong = norm.replace('pothole', 'crack')
    assert iou_reward(wrong, gt) == .2
    duplicate = json.dumps(json.loads(norm)*2)
    assert math.isclose(iou_reward(duplicate, gt), .6)
    print('Student code: labels, scale roundtrip, ordering and 6 reward cases passed.')
    verify_tensors()
    verify_quantization()
    verify_amp_overflow()


def verify_amp_overflow():
    import torch
    from kaggle_vlm import amp_optimizer_step
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=256.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    initial_epoch = scheduler.last_epoch
    # Simulate a real AMP overflow after backward, then retry with the same loss.
    for bad in (float('inf'), float('nan')):
        original = parameter.detach().clone()
        scale = scaler.get_scale()
        scaler.scale(parameter.square().sum()).backward()
        parameter.grad.fill_(bad)
        updated, norm = amp_optimizer_step(optimizer, scaler, [parameter])
        assert not updated and norm is None
        assert torch.equal(parameter, original)
        assert parameter.grad is None and scaler.get_scale() == scale*.5
        assert scheduler.last_epoch == initial_epoch
    scaler.scale(parameter.square().sum()).backward()
    updated, norm = amp_optimizer_step(optimizer, scaler, [parameter])
    assert updated and math.isclose(norm, 4.)
    scheduler.step()
    assert scheduler.last_epoch == initial_epoch+1
    assert math.isclose(parameter.item(), 1.9, abs_tol=1e-6)
    assert parameter.grad is None
    # The fp32 aggregate norm can overflow even if every gradient is finite.
    parameter = torch.nn.Parameter(torch.tensor([2., 2.]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=256.)
    original = parameter.detach().clone()
    scale = scaler.get_scale()
    scaler.scale(parameter.square().sum()).backward()
    parameter.grad.fill_(1e30)
    updated, _ = amp_optimizer_step(optimizer, scaler, [parameter])
    assert not updated and torch.equal(parameter, original)
    assert parameter.grad is None and scaler.get_scale() == scale*.5
    scaler.scale(parameter.square().sum()).backward()
    assert amp_optimizer_step(optimizer, scaler, [parameter])[0]
    print('Student code: AMP overflow skips bad updates, reduces scale and recovers on retry.')


def verify_quantization():
    import torch
    from kaggle_vlm import QUANTIZATION_SKIP_MODULES, validate_quantization
    # Match the skip rules in transformers 5.19 quantizers_utils.should_convert_module.
    def converts(name, patterns):
        return not any(re.match(f'{key}\\.', name) or re.match(key, name) or name.endswith(key)
                       for key in patterns)
    for name in ('model.visual.merger.linear_fc1',
                 'model.visual.deepstack_merger_list.0.linear_fc2',
                 'model.visual.blocks.0.attn.qkv', 'lm_head'):
        assert not converts(name, QUANTIZATION_SKIP_MODULES), name
    assert converts('model.visual.merger.linear_fc1', ['visual', 'lm_head'])
    assert converts('model.language_model.layers.0.self_attn.q_proj', QUANTIZATION_SKIP_MODULES)
    class Params4bit(torch.nn.Parameter): pass
    class Model:
        bad = False
        def named_parameters(self):
            yield 'model.visual.merger.linear_fc1.weight', (
                Params4bit(torch.ones(1)) if self.bad else torch.nn.Parameter(torch.ones(1)))
            yield 'model.language_model.layers.0.self_attn.q_proj.weight', Params4bit(torch.ones(1))
            yield 'lm_head.weight', torch.nn.Parameter(torch.ones(1))
    model = Model()
    validate_quantization(model)
    model.bad = True
    try: validate_quantization(model)
    except RuntimeError: pass
    else: raise AssertionError('Quantized vision weights must be rejected, even with a floating dtype')
    print('Student code: full-path quantization skips and early model validation passed.')


def verify_tensors():
    import torch
    from PIL import Image
    from kaggle_vlm import collate, encode_example
    short = {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long),
        'mm_token_type_ids': torch.tensor([[0, 1]]), 'labels': torch.tensor([[-100, 2]]),
        'pixel_values': torch.ones(3, 4), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    long = {'input_ids': torch.tensor([[3, 4, 5]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long),
        'mm_token_type_ids': torch.tensor([[0, 1, 0]]), 'labels': torch.tensor([[-100, -100, 5]]),
        'pixel_values': torch.ones(5, 4), 'image_grid_thw': torch.tensor([[1, 2, 4]])}
    batch = collate([short, long], pad_token_id=0)
    assert batch['input_ids'].tolist() == [[1, 2, 0], [3, 4, 5]]
    assert batch['labels'].tolist() == [[-100, 2, -100], [-100, -100, 5]]
    assert batch['attention_mask'].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch['mm_token_type_ids'].tolist() == [[0, 1, 0], [0, 1, 0]]
    assert batch['pixel_values'].shape == (8, 4)
    assert batch['image_grid_thw'].shape == (2, 3)
    class Processor:
        bad_prefix = False
        def apply_chat_template(self, messages, **kwargs):
            ids = [1, 2, 3] if len(messages) == 1 else [1, 2, 3, 4, 5]
            if len(messages) == 2 and self.bad_prefix: ids[0] = 99
            return {'input_ids': torch.tensor([ids])}
    Path('.pytest_cache').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir='.pytest_cache') as tmp:
        path = (Path(tmp) / 'image.jpg').resolve()
        Image.new('RGB', (640, 360)).save(path)
        row = {'image_path': str(path), 'width': 640, 'height': 360, 'objects': []}
        processor = Processor()
        encoded = encode_example(processor, row)
        assert encoded['labels'].tolist() == [[-100, -100, -100, 4, 5]]
        processor.bad_prefix = True
        try: encode_example(processor, row)
        except RuntimeError: pass
        else: raise AssertionError('Inconsistent chat prefix must fail')
    print('Student code: variable-length padding, vision concatenation and loss masking passed.')


if __name__ == '__main__':
    main()
