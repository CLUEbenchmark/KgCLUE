CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/chinese_rbt3_pytorch
export OUTPUR_DIR=$CURRENT_DIR/outputs
export DATA_DIR=../../datasets/ # 项目根目录
TASK_NAME="kg"


# 训练NER模型
# train(--do_train, --do_eval)
if [ $# == 0 ]; then
python run_ner_span.py \
  --do_train \
  --do_eval \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_adv \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=$DATA_DIR \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=6.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
# 在测试集上做预测 |run below lines to generate predicted file on test.json
elif [ $1 == "predict" ]; then
python run_ner_span.py \
  --do_predict  \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_adv \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=$DATA_DIR \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
 fi
