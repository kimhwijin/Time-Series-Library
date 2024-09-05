model_name=TimeMIL

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --root_path ./dataset/EthanolConcentration/ \
    --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --learning_rate 0.001 \
  --train_epochs 40 \
  --patience 10 \
  --d_model 128  \
  --dropout 0.2 \
  --use_gpu False \
  --num_workers 0 \
  --epoch_des 10