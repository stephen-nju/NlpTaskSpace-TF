 python -m task_compose.task_classification.albert_knowledge_distill.task_train_student \
 --train_data=data/train_data_v2/train.txt \
 --dev_data=data/train_data_v2/test.txt \
 --labels_name=data/train_data_v2/label.pickle \
 --labels_name_t=data/train_data/label.pickle \
 --teacher_model_path=output/model/albert_ml.h5 \
 --loss_type=asymmetric_loss \
 --output_root=output_v3 \
 --epochs=20
