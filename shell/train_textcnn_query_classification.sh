
export PROJECT_PATH="E:\NlpProgram\NlpTaskSpace-TF"

cd ${PROJECT_PATH}

export PYTHONPATH=${PROJECT_PATH}


python task_classification/textcnn/task_train_textcnn.py \
--train_data= \
--dev_data= \
