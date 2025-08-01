gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

coco_val_img=/path/to/coco/val2017/
coco_val_json=/path/to/coco/annotations/instances_val2017.json

CHUNKS=${#GPULIST[@]}

for CKPT in {"LMM-Det-stage4",}
do

for IDX in $(seq 0 $((CHUNKS-1))); do
   CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_coco_owlv2 \
       --model-path ./checkpoints/$CKPT \
       --answers-file ./playground/data/eval/coco/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
       --num-chunks $CHUNKS \
       --chunk-idx $IDX \
       --temperature 0 \
       --conv-mode vicuna_v2 \
       --image-folder $coco_val_img \
       --json-file $coco_val_json &
done
wait


output_file=./playground/data/eval/coco/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# # Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
	cat ./playground/data/eval/coco/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_coco_result.py --src $output_file --gt $coco_val_json

echo $CKPT

done


# CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_coco_owlv2 \
#        --model-path ./checkpoints/LMM-Det-stage4 \
#        --answers-file ./playground/data/eval/coco/answers/LMM-Det-stage4-test.jsonl \
#        --temperature 0 \
#        --conv-mode vicuna_v2 \
#        --image-folder $coco_val_img \
#        --json-file $coco_val_json