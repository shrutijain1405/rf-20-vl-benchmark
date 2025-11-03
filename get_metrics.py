import os
import json
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
from rf100vl.util import get_basename, get_category
import numpy as np

RESULT_DIR = '/home/spjain/qwen2.5_results_mat/results/Qwen2.5-VL-7B-Instruct_basic_serial/'
DATA_DIR = '/data3/spjain/rf20-vl-fsod'

all_dataset_dirs = {os.path.basename(d) : d for d in glob.glob(os.path.join(DATA_DIR, "*"))
                               if os.path.isdir(d) and os.path.exists(os.path.join(d, "test")) }

finals = defaultdict(list)
df = {}
tot = 0

for d in os.listdir(RESULT_DIR):
    if(d.startswith("evaluation") or d.endswith(".pkl")):
        continue
    dataset_name = (d.split("_")[1]).split(".")[0]
    dataset_dir = all_dataset_dirs[dataset_name]
    test_dir = os.path.join(dataset_dir, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    coco_gt = COCO(ann_path)
    predictions_path = os.path.join(
        RESULT_DIR, f"predictions_{dataset_name}.json"
    )
    with open(predictions_path, "r", encoding="utf-8") as f:
            detections_all = json.load(f)
    coco_dt = coco_gt.loadRes(detections_all)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save the evaluation stats right here
    stats = coco_eval.stats  # numpy array of length 12
    # Build a dictionary if you'd like more descriptive field names:
    stats_dict = {
        "AP_50_95": stats[0],
        "AP_50":    stats[1],
        "AP_75":    stats[2],
        "AP_small": stats[3],
        "AP_medium":stats[4],
        "AP_large": stats[5],
        "AR_1":     stats[6],
        "AR_10":    stats[7],
        "AR_100":   stats[8],
        "AR_small": stats[9],
        "AR_medium":stats[10],
        "AR_large": stats[11],
    }

    eval_results_path = os.path.join(
        RESULT_DIR, f"evaluation_{dataset_name}.json"
    )
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"Saved evaluation results to {eval_results_path}")

    score = stats_dict["AP_50_95"]*100    
    cat = get_category(dataset_name)
    finals[cat].append(score)
    df[dataset_name] = score
    tot += score

with open(RESULT_DIR+"evaluation_all_datasets.json", "w") as f:
    json.dump(df, f, indent=4)

output = []
for k, v in finals.items():
    avg = np.mean(v)
    print(f"{k}: {avg}")
    d = {}
    d[k] = avg
    output.append(d)

d = {}
d["all"] = tot/20
output.append(d)
with open(RESULT_DIR+"evaluation_per_category.json", "w") as f:
    json.dump(output, f, indent=4)





