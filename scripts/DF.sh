dataset=new_wf_L65536

python -u exp/train.py \
  --dataset ${dataset} \
  --model DF \
  --device cuda:0 \
  --feature DIR \
  --seq_len 131072 \
  --train_epochs 60 \
  --batch_size 2 \
  --num_workers 0 \
  --learning_rate 2e-3 \
  --optimizer Adamax \
  --lradj StepLR \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

# python -u exp/test.py \
#   --dataset ${dataset} \
#   --model DF \
#   --device cuda:0 \
#   --feature DIR \
#   --seq_len 131072 \
#   --batch_size 2 \
#   --eval_metrics Accuracy Precision Recall F1-score \
#   --load_name max_f1