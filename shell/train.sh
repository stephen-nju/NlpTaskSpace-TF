 python -m task_compose.task_classification.albert.task_albert_ml_classification \
 --train_data=data/train_data/train.txt \
 --dev_data=data/train_data/test.txt \
 --labels_name=data/train_data/label.pickle \
 --loss_type=asymmetric_loss \
 --output_root=output \
 --epochs=15
