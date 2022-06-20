python -m task_compose.deep_model_consolidation.task_train_old_model \
  --output_root=output_old \
  --train_data=data/train_data/train.txt \
  --dev_data=data/train_data/test.txt \
  --bert_model=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/albert_tiny_zh_google \
  --labels_name=data/train_data/label.pickle \
  --epochs=10 \
  --loss_type=focal_loss

