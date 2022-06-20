python -m task_compose.deep_model_consolidation.task_train_incremental_model \
  --output_root=output_incremental \
  --train_data=data/train_data_v2/train.txt \
  --dev_data=data/train_data_v2/test.txt \
  --bert_model=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/albert_tiny_zh_google \
  --labels_name=data/train_data_v2/label.pickle \
  --loss_type=focal_loss \
  --epochs=20

