# Added by Yang B.

import json
import os
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("clef_captioning")
class CLEFCaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, repetition_penalty=1.0):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric

        self.repetition_penalty = repetition_penalty
        print('Setting num_beams to', num_beams)
        print('Setting repetition_penalty to', repetition_penalty)

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        
        # Added by Yang B.
        repetition_penalty = float(run_cfg.get('repetition_penalty', 1.0))

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            repetition_penalty=repetition_penalty,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            repetition_penalty=self.repetition_penalty,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})
        print(results[-1])

        return results

    def after_evaluation(self, val_result, split_name, epoch, run_cfg, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name, run_cfg=run_cfg
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name, run_cfg):

        gt_root_or_file = getattr(run_cfg, f"{split_name}_gt_file")            
        coco_val = coco_caption_eval(gt_root_or_file, eval_result_file)

        # modified by Yang B.: add monitors
        monitors = getattr(run_cfg, 'monitors', ['R-1'])
        agg_metrics = sum([coco_val.eval[metric] for metric in monitors])
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        log_stats[split_name]['num_beams'] = self.num_beams
        log_stats[split_name]['repetition_penalty'] = self.repetition_penalty

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res



from pycocotools.coco import COCO


def coco_caption_eval(gt_root_or_file, results_file):
    # create coco object and coco_result object
    coco = COCO(gt_root_or_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval



from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import rouge


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self, scorers=None, calculate_naive_rouge=True):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        if scorers is None:
            print('setting up scorers...')
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                # (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
                # (Spice(), "SPICE"),
            ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

        if calculate_naive_rouge:
            for k, v in self.get_rouge_scores(res, gts).items():
                self.setEval(v, k)

        self.RES = res

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
    
    def get_rouge_scores(self, hyps, gts, scorer=rouge.Rouge()):
        self.HYPS = [c for k, v in hyps.items() for c in v]
        self.GTS = [c for k, v in gts.items() for c in v]
        
        scores = scorer.get_scores(self.HYPS, self.GTS, avg=True, ignore_empty=True)
        return {
            'R-1': scores['rouge-1']['f'],
            'R-2': scores['rouge-2']['f'],
            'R-L': scores['rouge-l']['f'],
        }
