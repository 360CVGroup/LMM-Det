import argparse
import torch
import os
import json
import math
import random
import glob
import numpy as np
import copy
from tqdm import tqdm
from PIL import Image

from llava.constants import COCO_80_CATEGORY, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, coco_val_all_imgs_path, tokenizer, image_processor, model_config):
        self.coco_val_all_imgs_path = coco_val_all_imgs_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        image_file = self.coco_val_all_imgs_path[index]
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        return image_tensor

    def __len__(self):
        return len(self.coco_val_all_imgs_path)


# DataLoader
def create_data_loader(coco_val_all_imgs_path, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(coco_val_all_imgs_path, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    coco_val_all_imgs_path = glob.glob(args.image_folder + "*")
    coco_val_all_imgs_path = sorted(coco_val_all_imgs_path)

    coco_val_all_imgs_path = get_chunk(coco_val_all_imgs_path, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(coco_val_all_imgs_path, tokenizer, image_processor, model.config)

    id2name_en = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90:'toothbrush'
            }
    enname2id = {v: k for k, v in id2name_en.items()}

    gt_jsonfile = args.json_file
    with open(gt_jsonfile, "r") as f:
        data = json.load(f)

    image_path_dict = {}
    image_path_w = {}
    image_path_h = {}
    for line in data["images"]:
        file_name = args.image_folder + line["file_name"]
        image_path_dict[file_name] = line["id"]
        image_path_w[file_name] = line["width"]
        image_path_h[file_name] = line["height"]

    num_all_bbox_class = 0
    pred_results = []

    for image_tensor, img_path in tqdm(zip(data_loader, coco_val_all_imgs_path), total=len(coco_val_all_imgs_path)):
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2

        
        for idx in range(0, len(COCO_80_CATEGORY)):
            all_bbox_class = []
            category1 = COCO_80_CATEGORY[idx]
            qs = "Detect all the objects in the image that belong to the category set {" + category1 + "}."
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
                
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0)

            input_ids = input_ids.to(device='cuda', non_blocking=True)

            with torch.inference_mode():
                outputs_all = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    num_beam_groups=args.num_beam_groups,
                    min_new_tokens=args.min_new_tokens,
                    max_new_tokens=4096,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            output_ids = outputs_all["sequences"] # 
            scores = outputs_all["scores"] # tuple, valid_token * 32k(vocabulary_size)

            if len(scores[0]) > 1:
                scores = [item[0].unsqueeze(0) for item in scores]

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            if "There are no objects in the image that belong to the required category set" in outputs:
                pass
            else:
                all_bbox_class = outputs.split(";")
                one_img_pred_results = get_coco_json(all_bbox_class, enname2id, category1, image_path_dict, image_path_w, image_path_h, img_path)
                pred_results += one_img_pred_results

    for line in pred_results:
        ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

    print(num_all_bbox_class)


def get_coco_json(all_bbox_class, enname2id, category1, image_path_dict, image_path_w, image_path_h, img_path):
    one_img_pred_results = []

    image_id = image_path_dict[img_path]
    width = image_path_w[img_path]
    height = image_path_h[img_path]

    for i, bbox_class in enumerate(all_bbox_class):
        try:
            bbox = eval(bbox_class.split("[")[1].split("],")[0].strip())
            assert bbox[0] <= 1
            assert bbox[1] <= 1
            assert bbox[2] <= 1
            assert bbox[3] <= 1
            class_id = enname2id[category1]
            class_score = eval(bbox_class.split("[")[1].split("],")[1].strip())
            bbox = ori_bbox_border_xywh(bbox, [width, height])

            one_img_pred_results.append({'image_id': image_id, 'category_id': class_id, 'bbox': bbox, 'score': class_score})
        except:
            if bbox_class:
                print(bbox_class)
            continue

    return one_img_pred_results


def ori_bbox_border_xywh(nor_bbox, img_size):
    w = img_size[0]
    h = img_size[1]
    big = max(h, w)
    border_w = border_h = 0

    x1 = nor_bbox[0] * big - border_w
    y1 = nor_bbox[1] * big - border_h
    x2 = nor_bbox[2] * big - border_w
    y2 = nor_bbox[3] * big - border_h
    return [x1, y1, x2-x1, y2-y1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--json-file", type=str, default="instances_val2017.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    args = parser.parse_args()

    eval_model(args)