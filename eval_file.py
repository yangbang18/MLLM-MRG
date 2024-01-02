import os
import torch
import json
import argparse
from tqdm import tqdm
from lavis.tasks.clef_captioning import coco_caption_eval

parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str)
parser.add_argument("--gt", type=str, default='./data/clef_2023_caption/val_gt.json')

parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument("-bs", "--bert_score", action='store_true')
parser.add_argument("--bert_score_model", type=str, default='./data/checkpoints/microsoft_deberta-xlarge-mnli')
parser.add_argument("--bert_score_num_layers", type=int, default=40)

parser.add_argument('-bleurt', '--bleurt', action='store_true')
parser.add_argument('--bleurt_model', type=str, default='./data/checkpoints/lucadiliello_BLEURT-20')

parser.add_argument('-cs', '--clip_score', action='store_true')
parser.add_argument('--clip_download_root', type=str, default='./data/checkpoints')

parser.add_argument('-ss', '--save_scores', action='store_true')
parser.add_argument('--save_path', type=str)
parser.add_argument('--save_filename', type=str, default='scores.json')
args = parser.parse_args()

if args.save_scores:
    assert args.save_path is not None, 'Please specify --save_path'
    os.makedirs(args.save_path, exist_ok=True)

coco_eval = coco_caption_eval(args.gt, args.pred)
eval_scores = coco_eval.eval

if args.bert_score:
    from bert_score import score
    P, R, F1 = score(
        cands=coco_eval.HYPS, 
        refs=coco_eval.GTS, 
        lang='en', 
        verbose=True,
        model_type=args.bert_score_model,
        num_layers=args.bert_score_num_layers,
        batch_size=args.batch_size,
    )
    eval_scores['BERTScore-P'] = P.mean().item()
    eval_scores['BERTScore-R'] = R.mean().item()
    eval_scores['BERTScore-F1'] = F1.mean().item()
    print(f"BERTScore -- P: {P.mean().item()}, R: {R.mean().item()}, F1: {F1.mean().item()}")

if args.bleurt:
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BleurtConfig.from_pretrained(args.bleurt_model)
    model = BleurtForSequenceClassification.from_pretrained(args.bleurt_model).to(device)
    tokenizer = BleurtTokenizer.from_pretrained(args.bleurt_model)

    references = coco_eval.GTS
    candidates = coco_eval.HYPS

    model.eval()
    n_batches = len(references) // args.batch_size
    if n_batches * args.batch_size != len(references):
        n_batches += 1

    results = []
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            refs = references[i*args.batch_size:(i+1)*args.batch_size]
            hyps = candidates[i*args.batch_size:(i+1)*args.batch_size]
            inputs = tokenizer(refs, hyps, padding=True, return_tensors='pt', truncation=True, max_length=512)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            res = model(**inputs).logits.cpu().flatten().tolist()
            results.extend(res)

    eval_scores['BLEURT'] = sum(results) / len(results)
    print('BLEURT:', eval_scores['BLEURT'])

if args.clip_score:
    from lavis.scorers import clipscore
    root = os.path.dirname(args.gt)
    val_data = json.load(open(os.path.join(root, 'val.json')))
    id2image = {}
    for line in val_data:
        id2image[line['image_id']] = line['image']
    candidates = []
    for k, v in coco_eval.RES.items():
        line = {'image': id2image[k], 'caption': v[0], 'image_id': k}
        candidates.append(line)
    eval_scores['CLIPScore'] = clipscore(candidates, image_dir=root, download_root=args.clip_download_root)

print('-'*10, 'evaluation scores')
for k, v in eval_scores.items():
    print(k, v)

if args.save_scores:
    with open(os.path.join(args.save_path, args.save_filename), 'w') as wf:
        json.dump(eval_scores, wf, indent=2)
