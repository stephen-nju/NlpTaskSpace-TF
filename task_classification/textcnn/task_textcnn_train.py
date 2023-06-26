
# -*- coding: utf-8 -*-
"""
@author: zhubin
"""

import itertools
import json
import argparse
import os
import random
import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from dependence.dataset import InputExampleBase, DataProcessorBase, InputFeaturesBase
from dependence.snippets import convert_to_unicode, sequence_padding
from tensorflow.keras.preprocessing.text import Tokenizer


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

logger=logging.getLogger(__name__)

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # tf gpu fix seed, please `pip install tensorflow-determinism` first
setup_seed(42)

class DataProcessorFunctions(DataProcessorBase):

    def get_train_examples(self, data_path):
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 20:
            #     break
            guid = "train-%d" % (i)
            text_a = convert_to_unicode(line[0])
            label = line[1].strip()
            examples.append(
                InputExampleBase(guid=guid,
                             text_a=text_a,
                             text_b=None,
                             label=label))
            if i % 20000 == 0:
                # 验证数据正确性
                logger.info(f"train data text_a:{text_a}")
                logger.info(f"train data label:{label}")
                
        return examples

    def get_dev_examples(self, data_path):
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 20:
            #     break
            guid = "train-%d" % (i)
            text_a = convert_to_unicode(line[0])
            label = line[1].strip()
            examples.append(
                InputExampleBase(guid=guid,
                             text_a=text_a,
                             text_b=None,
                             label=label))
        return examples

    @staticmethod
    def get_labels(path=None):
        return ["shouji", "other"]

    @staticmethod
    def build_tokenizer(examples,tokenizer,vocab_file):
        texts=[example.text_a for example in examples]
        tokenizer.fit_on_texts(texts)
        word_dict = tokenizer.word_index
        json.dump(word_dict, open(vocab_file, 'w',encoding="utf-8"), ensure_ascii=False)
        return tokenizer
        

    @staticmethod
    def convert_single_example(example, tokenizer, label_encode=None, max_length=128):
        
        input_ids =tokenizer.texts_to_sequences([example.text_a])
    
        # print(f"example={example.text_a}===input_ids={input_ids}")
        label_id = label_encode.transform([example.label])
        feature = InputFeaturesBase(
            guid=example.guid,
            input_ids=input_ids[0],
            input_mask=None,
            segment_ids=None,
            label_id=label_id,
            is_real_example=True)
        return feature

class DataSequence(Sequence):
    def __init__(self, features, token_pad_id, num_classes, batch_size,max_length):
        self.batch_size = batch_size
        self.features = features
        self.token_pad_id = token_pad_id
        self.num_classes = num_classes
        self.max_length=max_length
    def __len__(self):
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, index):
        data = self.features[index * self.batch_size:(index + 1) * self.batch_size]
        
        return self.feature_batch_transform(data)

    def feature_batch_transform(self, features):
        batch_token_ids,  batch_labels = [], []
        for feature in features:
            batch_token_ids.append(feature.input_ids)
            batch_labels.append(feature.label_id)
        batch_token_ids = sequence_padding(batch_token_ids,length=self.max_length,value=self.token_pad_id)
        
        return  batch_token_ids, to_categorical(np.array(batch_labels, dtype=float),
                                                                     num_classes=self.num_classes)


    def __call__(self, *args, **kwargs):
        return self


class ClassificationReporter(Callback):
    def __init__(self, validation_data):
        super(ClassificationReporter, self).__init__()
        self.validation_data = validation_data
        self.val_f1s = None
        self.val_recalls = None
        self.val_precisions = None

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        val_pred = []
        val_true = []
        for (x_val, y_val) in self.validation_data:
            val_pred_batch = self.model(x_val)
            # 这里可以设置不同的阈值进行验证
            val_pred.append(np.argmax(np.asarray(val_pred_batch).round(2), axis=1))
            val_true.append(np.argmax(np.asarray(y_val), axis=1))

        val_pred = np.asarray(list(itertools.chain.from_iterable(val_pred)))
        val_true = np.asarray(list(itertools.chain.from_iterable(val_true)))
        _val_f1 = f1_score(val_true, val_pred, average='macro')
        _val_recall = recall_score(val_true, val_pred, average='macro')
        _val_precision = precision_score(val_true, val_pred, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("\n")
        print(classification_report(val_true, val_pred, digits=4))
        return

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output", help="output dir")
    parse.add_argument("--train_data", type=str, default="data/textcnn/train.txt", help="train data path")
    parse.add_argument("--dev_data", type=str, default="data/textcnn/test.txt", help="validation data path")
    parse.add_argument("--lr", type=float, default=1e-5)
    parse.add_argument("--batch_size", type=int, default=4, help="batch size")
    parse.add_argument("--embedding_dim",type=int,default=300,help="word embedding dim")
    parse.add_argument("--smoothing", type=float, default=0.1, help="batch size")
    parse.add_argument("--max_length", type=int, default=64, help="max sequence length")
    parse.add_argument("--epochs", type=int, default=2, help="number of training epoch")
    parse.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal_loss"],
                       help="use bce for binary cross entropy loss and focal for focal loss")
    # 参数配置
    args = parse.parse_args()
    # tokenizer
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    label_encode = LabelEncoder()
    # 模型
    processor = DataProcessorFunctions()

    # 设置类别标签
    labels_raw = processor.get_labels()
    label_encode.fit(labels_raw)
    # 二维列表

    call_backs = []
    log_dir = os.path.join(args.output_root, "logs")
    if os.path.exists(log_dir):
        # 清除日志
        logger.info(f"remove log before in {log_dir} ")
    else:
        os.makedirs(log_dir)
    
    tokenizer=Tokenizer(oov_token="<UNK>")
    #######################################################################
    train_examples = processor.get_train_examples(args.train_data)
    dev_examples = processor.get_dev_examples(args.dev_data)
    # 构建词典
    total_example=train_examples+dev_examples
    
    tokenizer=processor.build_tokenizer(total_example,tokenizer,"vocab.txt")
    

    train_features = processor.convert_examples_to_features(train_examples,
                                                            tokenizer=tokenizer,
                                                            max_length=args.max_length,
                                                            label_encode=label_encode)


    dev_features = processor.convert_examples_to_features(dev_examples,
                                                          tokenizer=tokenizer,
                                                          max_length=args.max_length,
                                                          label_encode=label_encode)


    train_sequence = DataSequence(train_features,
                                  token_pad_id=0,
                                  num_classes=len(label_encode.classes_),
                                  batch_size=args.batch_size,
                                  max_length=args.max_length)
    # 加载验证集数据   
    dev_sequence = DataSequence(dev_features,
                                token_pad_id=0,
                                num_classes=len(label_encode.classes_),
                                batch_size=args.batch_size,
                                max_length=args.max_length)

    #####添加callback#########################################################
    report = ClassificationReporter(validation_data=dev_sequence)
    call_backs.append(report)
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=2)
    call_backs.append(tensor_board)
    model_dir = os.path.join(args.output_root, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'roberta_ml.h5'), monitor='loss', verbose=2,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)
    call_backs.append(checkpoint)
    early_stop = EarlyStopping('loss', patience=4, mode='min', verbose=2, restore_best_weights=True)
    call_backs.append(early_stop)

    ##################构建模型################################
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"{strategy.num_replicas_in_sync} number of devices")
    
    ########注意词典大小################
    vocab_size=len(tokenizer.word_index)+1
    print(f"vocab_size==={vocab_size}")
    # 卷积和大小
    filter_sizes=[3,4,5]
    with strategy.scope():
        inputs = tf.keras.Input(shape=(args.max_length,), dtype="int64")
        
        x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=args.embedding_dim,input_length=args.max_length)(inputs)
        x =tf.keras.layers.Reshape((args.max_length, args.embedding_dim, 1))(x)
        
        cnns = []
        for size in filter_sizes:
            conv =tf.keras.layers.Conv2D(filters=64, kernel_size=(size, args.embedding_dim),
                                strides=(1,1), padding='valid', activation='relu')(x)
            pool = tf.keras.layers.MaxPooling2D(pool_size=(args.max_length-size+ 1, 1), strides=(1,1),padding='valid')(conv)
            
            cnns.append(pool)

        x = tf.keras.layers.concatenate(cnns,axis=-1)
        x =tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
    
        x=tf.keras.layers.Dropout(0.2)(x)

        outputs=tf.keras.layers.Dense(2, activation='softmax',name="output")(x)
        # 
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        
        lr_schedule = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=args.lr,
            maximal_learning_rate=3e-5,
            step_size=800,
            scale_fn=lambda x: 1.,
            scale_mode="cycle",
            name="MyCyclicScheduler")

        # #######构建loss和metric：
        if args.loss_type == "ce":
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                       label_smoothing=args.smoothing,
                                                                       reduction=tf.keras.losses.Reduction.NONE,
                                                                       ),
                          # metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()],
                          optimizer=Adam(learning_rate=args.lr))
        if args.loss_type == "focal_loss":
            model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,
                                                                   reduction=tf.keras.losses.Reduction.NONE
                                                                   ),
                          optimizer=lr_schedule
                          )
    model.summary()
    model.fit(train_sequence,
              batch_size=args.batch_size,
              validation_data=dev_sequence,
              steps_per_epoch=len(train_sequence),
              epochs=args.epochs,
              # use_multiprocessing=False,
              # train_sequence注意使用可序列化的对象
              callbacks=call_backs,
              # 设置分类权重
              )
