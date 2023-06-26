# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_train_param_cleaning.py
@time: 2022/3/10 13:52
"""
import argparse
import itertools
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from loguru import logger
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from transformers.models.bert import BertConfig, BertTokenizer, TFBertForSequenceClassification
from dependence.backend import search_layer
from dependence.dataset import InputExampleBase, DataProcessorBase, InputFeaturesBase
from dependence.snippets import convert_to_unicode, sequence_padding


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)


class InputExample(InputExampleBase):

    def __init__(self, guid, text_a, text_b=None, text_c=None, text_d=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_c = text_c
        self.text_d = text_d
        super().__init__(guid, text_a, text_b, label)


class DataProcessorFunctions(DataProcessorBase):

    def get_train_examples(self, data_path):
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 20:
            #     break
            guid = "train-%d" % (i)
            text_a = convert_to_unicode(line[0])
            text_b = convert_to_unicode(line[1])
            text_c = convert_to_unicode(line[2])
            text_d = convert_to_unicode(line[3])
            label = line[4].strip()
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             text_c=text_c,
                             text_d=text_d,
                             label=label))
            if i % 20000 == 0:
                # 验证数据正确性
                logger.info(f"train data text_a:{text_a}")
                logger.info(f"train data text_b:{text_b}")
                logger.info(f"train data text_c:{text_c}")
                logger.info(f"train data text_d:{text_d}")
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
            text_b = convert_to_unicode(line[1])
            text_c = convert_to_unicode(line[2])
            text_d = convert_to_unicode(line[3])
            label = line[4].strip()
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             text_c=text_c,
                             text_d=text_d,
                             label=label))
        return examples

    def get_test_examples(self, data_path):
        pass

    @staticmethod
    def get_labels(path=None):
        return ["positive", "negative"]

    @staticmethod
    def convert_single_example(example, tokenizer, label_encode=None, max_length=128):
        encoder = tokenizer("[SEP]".join([example.text_a, example.text_b, example.text_c, example.text_d]),
                            max_length=max_length, truncation=True)
        input_ids = encoder.input_ids
        segment_ids = encoder.token_type_ids
        input_mask = encoder.attention_mask
        label_id = label_encode.transform([example.label])
        feature = InputFeaturesBase(
            guid=example.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature


class DataSequence(Sequence):
    def __init__(self, features, token_pad_id, num_classes, batch_size):
        self.batch_size = batch_size
        self.features = features
        self.token_pad_id = token_pad_id
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, index):
        data = self.features[index * self.batch_size:(index + 1) * self.batch_size]
        return self.feature_batch_transform(data)

    def feature_batch_transform(self, features):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for feature in features:
            batch_token_ids.append(feature.input_ids)
            batch_segment_ids.append(feature.segment_ids)
            batch_labels.append(feature.label_id)

        batch_token_ids = sequence_padding(batch_token_ids, value=self.token_pad_id)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        return {"input_ids": batch_token_ids,
                "token_type_ids": batch_segment_ids}, to_categorical(np.array(batch_labels, dtype=float),
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
            val_pred_batch = tf.nn.softmax(self.model(x_val).logits)
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
        print(classification_report(val_true, val_pred, digits=4))
        return


# def adversarial_training(model, embedding_name, epsilon=1):
#     """给模型添加对抗训练
#     其中model是需要添加对抗训练的keras模型，embedding_name
#     则是model里边Embedding层的名字。要在模型compile之后使用。
#     """
#     if model.train_function is None:  # 如果还没有训练函数
#         model._make_train_function()  # 手动make
#     old_train_function = model.train_function  # 备份旧的训练函数
#
#     # 查找Embedding层
#     for output in model.outputs:
#         embedding_layer = search_layer(output, embedding_name)
#         if embedding_layer is not None:
#             break
#     if embedding_layer is None:
#         raise Exception('Embedding layer not found')
#
#     # 求Embedding梯度
#     embeddings = embedding_layer.embeddings  # Embedding矩阵
#     gradients = tf.keras.gradients(model.total_loss, [embeddings])  # Embedding梯度
#     gradients = tf.keras.zeros_like(embeddings) + gradients[0]  # 转为dense tensor
#
#     # 封装为函数
#     inputs = (
#             model._feed_inputs + model._feed_targets + model._feed_sample_weights
#     )  # 所有输入层
#     embedding_gradients = tf.keras.function(
#         inputs=inputs,
#         outputs=[gradients],
#         name='embedding_gradients',
#     )  # 封装为函数

    # def train_function(inputs):  # 重新定义训练函数
    #     grads = embedding_gradients(inputs)[0]  # Embedding梯度
    #     delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
    #     tf.keras.set_value(embeddings, tf.keras.eval(embeddings) + delta)  # 注入扰动
    #     outputs = old_train_function(inputs)  # 梯度下降
    #     tf.keras.set_value(embeddings, tf.keras.eval(embeddings) - delta)  # 删除扰动
    #     return outputs
    #
    # model.train_function = train_function  # 覆盖原训练函数
    #

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output", help="output dir")
    parse.add_argument("--train_data", type=str, default="data/param_cleaning/train.txt", help="train data path")
    parse.add_argument("--dev_data", type=str, default="data/param_cleaning/train.txt", help="validation data path")
    parse.add_argument("--bert_model", type=str, default="E:\\Resources\\chinese-roberta-wwm-ext")
    parse.add_argument("--lr", type=float, default=1e-5)
    parse.add_argument("--batch_size", type=int, default=8, help="batch size")
    parse.add_argument("--smoothing", type=float, default=0.1, help="batch size")
    parse.add_argument("--max_length", type=int, default=128, help="max sequence length")
    parse.add_argument("--epochs", type=int, default=2, help="number of training epoch")
    parse.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal_loss"],
                       help="use bce for binary cross entropy loss and focal for focal loss")
    # 参数配置
    args = parse.parse_args()
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    label_encode = LabelEncoder()
    # 模型
    word_dict = tokenizer.get_vocab()
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
        os.removedirs(log_dir)
    else:
        os.makedirs(log_dir)

    bert_config = os.path.join(args.bert_model, "config.json")
    config = BertConfig.from_pretrained(bert_config)
    # 更新基本参数
    config.num_labels = len(label_encode.classes_)
    config.return_dict = True
    config.output_attentions = False
    config.output_hidden_states = False
    config.max_length = args.max_length

    #######################################################################
    train_examples = processor.get_train_examples(args.train_data)
    train_features = processor.convert_examples_to_features(train_examples,
                                                            tokenizer=tokenizer,
                                                            max_length=args.max_length,
                                                            label_encode=label_encode)

    # 使用tensorflow dataset
    # ds = tf.data.Dataset.from_generator(lambda: [asdict(a) for a in train_features],
    #                                     output_signature={
    #                                         "input_ids": tf.TensorSpec(shape=(), dtype=tf.int32),
    #                                         "input_mask": tf.TensorSpec(shape=(), dtype=tf.int32),
    #                                         "segment_ids": tf.TensorSpec(shape=(), dtype=tf.int32),
    #                                         "label_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
    #                                         "is_real_example": tf.TensorSpec(shape=(), dtype=tf.bool),
    #                                     }
    #                                     )
    #
    # ds = ds.batch(args.batch_size)
    # ds = ds.prefetch(10)
    # ds = ds.repeat()
    # for d in ds:
    #     print(d)
    train_sequence = DataSequence(train_features,
                                  token_pad_id=tokenizer.pad_token_id,
                                  num_classes=len(label_encode.classes_),
                                  batch_size=args.batch_size)
    # 加载验证集数据
    dev_examples = processor.get_dev_examples(args.dev_data)
    dev_features = processor.convert_examples_to_features(dev_examples,
                                                          tokenizer=tokenizer,
                                                          max_length=args.max_length,
                                                          label_encode=label_encode)
    dev_sequence = DataSequence(dev_features,
                                token_pad_id=tokenizer.pad_token_id,
                                num_classes=len(label_encode.classes_),
                                batch_size=args.batch_size)

    # #####添加callback#########################################################
    report = ClassificationReporter(validation_data=dev_sequence)
    call_backs.append(report)
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=2)
    call_backs.append(tensor_board)
    model_dir = os.path.join(args.output_root, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # os.removedirs()
    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'roberta_ml.h5'), monitor='loss', verbose=2,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)
    call_backs.append(checkpoint)
    early_stop = EarlyStopping('loss', patience=4, mode='min', verbose=2, restore_best_weights=True)
    call_backs.append(early_stop)

    # ##################构建模型################################
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"{strategy.num_replicas_in_sync} number of devices")
    with strategy.scope():
        model = TFBertForSequenceClassification.from_pretrained(args.bert_model,
                                                                config=config
                                                                )
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
    # model.compile(
    #     loss=model.compute_loss,
    #     optimizer=Adam(learning_rate=lr_schedule)
    # )
    model.summary()
    # #########模型训练######################################################
    # 新版本接口
    # ds = tf.data.Dataset.from_generator(train_sequence,
    #
    #                                     output_signature=(
    #                                         {"input_ids": tf.TensorSpec(shape=(None, None),
    #                                                                     dtype=tf.int32),
    #                                          "token_type_ids": tf.TensorSpec(shape=(None, None),
    #                                                                          dtype=tf.int32)
    #                                          },
    #                                         tf.TensorSpec(shape=(None, None),
    #                                                       dtype=tf.int32))
    #                                     )
    # ds_dev = tf.data.Dataset.from_generator(dev_sequence,
    #                                         output_signature=(
    #                                             {"input_ids": tf.TensorSpec(shape=(None, None),
    #                                                                         dtype=tf.int32),
    #                                              "token_type_ids": tf.TensorSpec(shape=(None, None),
    #                                                                              dtype=tf.int32)
    #                                              },
    #                                             tf.TensorSpec(shape=(None, None),
    #                                                           dtype=tf.int32))
    #                                         )
    # 老版本接口
    #
    # tf.data.Dataset.map()
    # ds = tf.data.Dataset.from_generator(train_sequence,
    #                                     output_types=({"input_ids": tf.int32, "token_type_ids": tf.int32},
    #                                                   tf.int32
    #                                                   ),
    #                                     output_shapes=({"input_ids": tf.TensorShape([None, None]),
    #                                                     "token_type_ids": tf.TensorShape([None, None])},
    #                                                    tf.TensorShape([None, None])
    #                                                    )
    #                                     )
    # ds_dev = tf.data.Dataset.from_generator(dev_sequence,
    #                                         output_types=({"input_ids": tf.int32, "token_type_ids": tf.int32},
    #                                                       tf.int32
    #                                                       ),
    #                                         output_shapes=({"input_ids": tf.TensorShape([None, None]),
    #                                                         "token_type_ids": tf.TensorShape([None, None])},
    #                                                        tf.TensorShape([None, None])
    #                                                        )
    # #                                         )
    # ds_train = ds.prefetch(100).repeat(args.epochs)

    # adversarial_training(model, 'Embedding-Token', 0.5)
    model.fit(train_sequence,
              validation_data=dev_sequence,
              steps_per_epoch=len(train_sequence),
              epochs=args.epochs,
              # use_multiprocessing=False,
              # train_sequence注意使用可序列化的对象
              callbacks=call_backs,
              # 设置分类权重
              )
