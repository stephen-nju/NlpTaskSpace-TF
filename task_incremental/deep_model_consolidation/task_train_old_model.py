# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_train_old_model.py
@time: 2022/2/10 11:26
"""
# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_train_teachers.py
@time: 2022/2/8 16:05
"""
import argparse
import csv
import itertools
import json
import os
import pickle
import random
from abc import ABC
from typing import Optional, Union, Tuple

import mlflow.keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tokenizers.implementations import BertWordPieceTokenizer
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.modeling_tf_utils import get_initializer, TFModelInputType, input_processing
from transformers.models.albert import AlbertConfig, TFAlbertPreTrainedModel
from transformers.models.albert.modeling_tf_albert import TFAlbertMainLayer

from dependence.lossing import AsymmetricLoss
from dependence.snippets import convert_to_unicode, sequence_padding

mlflow.keras.autolog()


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessorFunctions(object):

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_path):
        """See base class."""
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 20:
            #     break
            guid = "train-%d" % (i)
            text_a = convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=line[1]))
        return examples

    def get_test_examples(self, data_path):
        pass

    def get_dev_examples(self, data_path):
        """See base class."""
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 20:
            #     break
            guid = "dev-%d" % (i)
            text_a = convert_to_unicode(line[0])
            label = convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    @staticmethod
    def get_labels(label_path):
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
        return labels
        # return ["男", "女", "婴幼儿", "儿童", "少年", "青年", "中年", "老年", "学生", "孕妇", "上班族", "宠物用品"]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                assert len(line) == 3, print(line)
                data = json.loads(line[2])
                labels = []
                for key, value in data.items():
                    for v in value:
                        labels.append(":".join([key, v]))
                lines.append([line[1], labels])
            return lines

    @staticmethod
    def convert_single_example(example, tokenizer, label_encode):
        encoder = tokenizer.encode(sequence=example.text_a)
        input_ids = encoder.ids
        segment_ids = encoder.type_ids
        input_mask = encoder.attention_mask
        label_id = label_encode.transform([example.label])
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature

    def convert_examples_to_features(self, examples, tokenizer, label_encode):
        features = []
        # 这里可以采用多线程等方式提高预处理速度
        for example in examples:
            feature = self.convert_single_example(example=example, tokenizer=tokenizer, label_encode=label_encode)
            features.append(feature)

        return features


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
                "token_type_ids": batch_segment_ids}, np.squeeze(np.array(batch_labels, dtype=float),
                                                                 axis=1)


#
#
# def compute_focal_loss(self, labels, logits):
#     loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
#         from_logits=False,
#         reduction=tf.keras.losses.Reduction.NONE
#     )
#     return loss_fn(labels, logits)
#
#
# def compute_asymmetric_loss(self, labels, logits):
#     loss_fn = AsymmetricLoss(
#         from_logits=False,
#         gamma_pos=0,
#         gamma_neg=2,
#         clip=0.1,
#         reduction=tf.keras.losses.Reduction.NONE
#     )
#
#     return loss_fn(labels, logits)

# class TFMultiLabelClassificationLoss:
#     """
#     Loss function suitable for sequence classification.
#     """
#
#     def compute_loss(self, labels, logits):
#         loss_fn = tf.keras.losses.BinaryCrossentropy(
#             from_logits=False, reduction=tf.keras.losses.Reduction.NONE
#         )
#
#         return loss_fn(labels, logits)

# def compute_focal_loss(self, labels, logits):
#     loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
#         from_logits=False,
#         reduction=tf.keras.losses.Reduction.NONE
#     )
#     return loss_fn(labels, logits)
#
# def compute_asymmetric_loss(self, labels, logits):
#     loss_fn = AsymmetricLoss(
#         from_logits=False,
#         gamma_pos=0,
#         gamma_neg=2,
#         clip=0.1,
#         reduction=tf.keras.losses.Reduction.NONE
#     )
#
#     return loss_fn(labels, logits)


class TFAlbertForMultiLabelClassification(TFAlbertPreTrainedModel, ABC):
    _keys_to_ignore_on_load_unexpected = [r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier",
            activation="sigmoid"
        )

    def call(
            self,
            input_ids: Optional[TFModelInputType] = None,
            attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
            position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
            head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
            inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
            training: Optional[bool] = False,
            **kwargs,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.albert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=inputs["training"])
        logits = self.classifier(inputs=pooled_output)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return output

        return TFSequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


class ClassificationReporter(Callback):
    def __init__(self, validation_data):
        super(ClassificationReporter, self).__init__()
        self.validation_data = validation_data
        self.val_f1s = None
        self.val_recalls = None
        self.val_precisions = None

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = []
        val_true = []
        for (x_val, y_val) in self.validation_data:
            val_pred_batch = self.model.predict_on_batch(x_val).logits
            # 这里可以设置不同的阈值进行验证
            val_pred.append(np.asarray(val_pred_batch).round(2))
            val_true.append(np.asarray(y_val))
        val_pred = np.asarray(list(itertools.chain.from_iterable(val_pred)))
        val_pred = np.where(val_pred < 0.5, 0, 1)
        val_true = np.asarray(list(itertools.chain.from_iterable(val_true)))

        _val_f1 = f1_score(val_true, val_pred, average='macro')
        _val_recall = recall_score(val_true, val_pred, average='macro')
        _val_precision = precision_score(val_true, val_pred, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(classification_report(val_true, val_pred, digits=4))
        return


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output", help="output dir")
    parse.add_argument("--train_data", type=str, default="data/train_data/train.txt", help="train data path")
    parse.add_argument("--dev_data", type=str, default="data/train_data/test.txt", help="validation data path")
    parse.add_argument("--labels_name", type=str, default="data/train_data/label.pickle",
                       help="labels names in data")
    parse.add_argument("--albert", type=str, default="BERT", choices=["BERT", "ALBERT"], help="bert type")
    parse.add_argument("--bert_model", type=str, default="E:\\resources\\albert_tiny_zh_google")
    parse.add_argument("--lr", type=float, default=1e-5)
    parse.add_argument("--batch_size", type=int, default=8, help="batch size")
    parse.add_argument("--max_length", type=int, default=128, help="max sequence length")
    parse.add_argument("--epochs", type=int, default=10, help="number of training epoch")
    parse.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal_loss", "asymmetric_loss"],
                       help="use bce for binary cross entropy loss and focal for focal loss")
    # 参数配置
    args = parse.parse_args()
    # tokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_model, "vocab.txt"), pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=args.max_length)

    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    label_encode = MultiLabelBinarizer()
    # 模型
    word_dict = tokenizer.get_vocab()
    tokenizer.enable_truncation(max_length=args.max_length)
    processor = DataProcessorFunctions()
    # 设置类别标签
    labels_raw = processor.get_labels(args.labels_name)
    label_encode.fit([labels_raw])
    # 二维列表
    train_examples = processor.get_train_examples(args.train_data)
    train_features = processor.convert_examples_to_features(train_examples,
                                                            tokenizer=tokenizer,
                                                            label_encode=label_encode)
    train_sequence = DataSequence(train_features,
                                  token_pad_id=tokenizer.token_to_id("[PAD]"),
                                  num_classes=len(label_encode.classes_),
                                  batch_size=args.batch_size)
    # 加载验证集数据
    dev_examples = processor.get_dev_examples(args.dev_data)
    dev_features = processor.convert_examples_to_features(dev_examples,
                                                          tokenizer=tokenizer,
                                                          label_encode=label_encode)
    dev_sequence = DataSequence(dev_features,
                                token_pad_id=tokenizer.token_to_id("[PAD]"),
                                num_classes=len(label_encode.classes_),
                                batch_size=args.batch_size)

    call_backs = []
    log_dir = os.path.join(args.output_root, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    albert_config = os.path.join(args.bert_model, "config.json")
    config = AlbertConfig.from_pretrained(albert_config)
    # 更新基本参数
    config.num_labels = len(label_encode.classes_)
    config.return_dict = True
    config.output_attentions = False
    config.output_hidden_states = False
    config.max_length = args.max_length

    # ##################构建模型################################

    model = TFAlbertForMultiLabelClassification.from_pretrained(args.bert_model,
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
    if args.loss_type == "bce":
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                              reduction=tf.keras.losses.Reduction.SUM),
                      optimizer=Adam(learning_rate=lr_schedule),
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.AUC()])
    if args.loss_type == "focal_loss":
        model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False,
                                                               reduction=tf.keras.losses.Reduction.SUM),
                      optimizer=Adam(learning_rate=lr_schedule),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    if args.loss_type == "asymmetric_loss":
        model.compile(loss=AsymmetricLoss(from_logits=False,
                                          reduction=tf.keras.losses.Reduction.SUM),
                      optimizer=Adam(learning_rate=lr_schedule),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                      )
    model.summary()
    # ######添加callback#########################################################
    report = ClassificationReporter(validation_data=dev_sequence)
    call_backs.append(report)
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=True)
    call_backs.append(tensor_board)
    model_dir = os.path.join(args.output_root, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'albert_ml.h5'), monitor='binary_accuracy', verbose=2,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    call_backs.append(checkpoint)
    early_stop = EarlyStopping('binary_accuracy', patience=4, mode='max', verbose=2, restore_best_weights=True)
    call_backs.append(early_stop)

    # #########模型训练######################################################

    model.fit(train_sequence,
              validation_data=dev_sequence,
              steps_per_epoch=len(train_sequence),
              epochs=args.epochs,
              # use_multiprocessing=False,
              # train_sequence注意使用可序列化的对象
              callbacks=call_backs,
              # 设置分类权重
              )
