dataset=new_wf_L65536

python -u exp/train.py \
  --dataset ${dataset} \
  --model AWF \
  --device cuda:0 \
  --feature DIR \
  --seq_len 32768 \
  --train_epochs 50 \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset ${dataset} \
  --model AWF \
  --device cuda:0 \
  --feature DIR \
  --seq_len 32768 \
  --batch_size 2 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1