python -m task_compose.deep_model_consolidationt.task_consolidation_train \
  --output_root=output_v4 \
  --dev_data=data/train_data_v2/test.txt \
  --bert_model=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/albert_tiny_zh_google \
  --old_teacher_model_path=output/model/albert_ml.h5 \
  --incremental_teacher_model_path=output_v2/model/albert_ml.h5 \
