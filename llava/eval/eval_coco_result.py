
from pycocotools.coco import COCO
from cocoeval import COCOeval
import argparse
import json


def eval_coco_result(pred_results, gt_jsonfile):
    print(gt_jsonfile)
    coco_eval = (
        _evaluate_predictions_on_coco(
            COCO(gt_jsonfile),
            pred_results,
            'bbox',
            img_ids=None,
        )
        if len(pred_results) > 0
        else None  # cocoapi does not handle empty results very well
    )



def _evaluate_predictions_on_coco(
    coco_gt, coco_results, iou_type, img_ids=None
):
    """
    Evaluate the coco results using COCOEval API.
    """
    for i, d in enumerate(coco_results):
        if isinstance(d['score'], list):
            print(coco_results[i]['score'])
            coco_results[i]['score'] = coco_results[i]['score'][0]
    
    assert len(coco_results) > 0
    # coco_results follow xywh format
    coco_dt = coco_gt.loadRes(coco_results)
    # coco_eval = COCOeval_opt(coco_gt, coco_dt, iou_type)

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="facebook/opt-350m")
    parser.add_argument("--gt", type=str, default="coco/annotations/instances_val2017.json")
    args = parser.parse_args()
    
    pred_results = []
    for line in open(args.src):
        res = json.loads(line)
        pred_results.append(res)

    eval_coco_result(pred_results, args.gt)