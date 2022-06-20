# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_train_student.py
@time: 2022/1/18 16:48
"""
import argparse
import csv
import json
import logging
import os
import pickle
import random
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils import tf_utils
from tokenizers.implementations import BertWordPieceTokenizer
from transformers.file_utils import ModelOutput
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.modeling_tf_utils import get_initializer, TFModelInputType, input_processing
from transformers.models.albert import AlbertConfig, TFAlbertPreTrainedModel, TFAlbertMainLayer

from dependence.lossing import AsymmetricLoss
from dependence.snippets import convert_to_unicode, sequence_padding


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)


# tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)


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
            if i > 20:
               break
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
            #    break
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
        # 删除batch_size 不足的数据
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, index):
        data = self.features[index * self.batch_size:(index + 1) * self.batch_size]
        # print(index, len(data))
        return self.feature_batch_transform(data)

    def feature_batch_transform(self, features):
        batch_token_ids, batch_segment_ids, batch_attention_mask_ids, batch_labels = [], [], [], []
        for feature in features:
            batch_token_ids.append(feature.input_ids)
            batch_segment_ids.append(feature.segment_ids)
            batch_attention_mask_ids.append(feature.input_mask)
            batch_labels.append(feature.label_id)

        batch_token_ids = sequence_padding(batch_token_ids, value=self.token_pad_id)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        batch_attention_mask_ids = sequence_padding(batch_attention_mask_ids)
        return [batch_token_ids, batch_segment_ids], np.squeeze(
            np.array(batch_labels, dtype=float), axis=1)


class TFMultiLabelClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """

    def compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )

        return loss_fn(labels, logits)

    def compute_focal_loss(self, labels, logits):
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
        return loss_fn(labels, logits)


#
# class TFAlbertDistillLoss:
#     """
#     Loss function suitable for sequence classification.
#     """
#
#     def compute_loss(self, labels, logits):
#         print("label_shape", labels.shape)
#         # print("logits shape", logits[0].shape)
#         # print("logits shape2", logits[1].shape)
#         loss_fn = tf.keras.losses.BinaryCrossentropy(
#             from_logits=False, reduction=tf.keras.losses.Reduction.NONE
#         )
#         # teacher_loss_fn = tf.keras.losses.BinaryCrossentropy(
#         #     from_logits=True, reduction=tf.keras.losses.Reduction.NONE
#         # )
#         return loss_fn(labels, logits)
#
#     def compute_teacher_loss(self, teacher, student):
#         teacher_loss_fn = tf.keras.losses.BinaryCrossentropy(
#             from_logits=True, reduction=tf.keras.losses.Reduction.NONE
#         )
#         return teacher_loss_fn(teacher, student)
#
#     def compute_focal_loss(self, labels, logits):
#         loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
#             from_logits=False,
#             reduction=tf.keras.losses.Reduction.NONE
#         )
#         teacher_loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
#             from_logits=True,
#             reduction=tf.keras.losses.Reduction.NONE
#         )
#         return loss_fn(labels, logits)
#

class TeacherTFAlbertForMLC(TFAlbertPreTrainedModel, TFMultiLabelClassificationLoss):
    _keys_to_ignore_on_load_unexpected = [r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
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
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification.serving_output
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


@dataclass
class StudentTFAlbertOutput(ModelOutput):
    logits: Tuple[tf.Tensor, tf.Tensor] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


class StudentTFAlbertForMLC(TFAlbertPreTrainedModel, ABC):
    # def prune_heads(self, heads_to_prune):
    #     pass

    _keys_to_ignore_on_load_unexpected = [r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.student_num_labels = config.teacher_num_labels
        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.student_classifier = tf.keras.layers.Dense(units=self.student_num_labels,
                                                        kernel_initializer=get_initializer(config.initializer_range),
                                                        name="student_classifier")
        self.classifier = tf.keras.layers.Dense(units=self.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name="classifier",
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
    ) -> Union[StudentTFAlbertOutput, Tuple[tf.Tensor]]:
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

        student_logits = self.student_classifier(inputs=pooled_output)
        out_logits = self.classifier(inputs=pooled_output)

        # loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"],
        #                                                                logits=out_logits)
        # loss = None
        # if inputs["labels"] is not None:
        #     loss = self.compute_loss(labels=inputs["labels"], logits=out_logits)

        if not inputs["return_dict"]:
            output = (out_logits, student_logits) + outputs[2:]
            return output
            # return ((loss,) + output) if loss is not None else output

        return StudentTFAlbertOutput(
            # loss=loss,
            logits=(out_logits, student_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification.serving_output
    # def serving_output(self, output: StudentTFAlbertOutput) -> StudentTFAlbertOutput:
    #     hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
    #     attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
    #     return StudentTFAlbertOutput(logits=output.logits,
    #                                  hidden_states=hs,
    #                                  attentions=attns)


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
            val_pred_batch = self.model.predict_on_batch(x_val)

            # 这里可以设置不同的阈值进行验证
            val_pred.append(np.asarray(val_pred_batch).round(2))
            val_true.append(np.asarray(y_val))
        val_pred = np.squeeze(np.asarray(val_pred), axis=1)
        val_pred = np.where(val_pred < 0.5, 0, 1)
        val_true = np.squeeze(np.asarray(val_true), axis=1)
        _val_f1 = f1_score(val_true, val_pred, average='macro')
        _val_recall = recall_score(val_true, val_pred, average='macro')
        _val_precision = precision_score(val_true, val_pred, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(classification_report(val_true, val_pred, digits=4))
        return


class StudentModelCheckpoint(ModelCheckpoint):
    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.student.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.student.save(filepath, overwrite=True, options=self._options)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.student.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.student.save(filepath, overwrite=True, options=self._options)

                self._maybe_remove_file()
            except IsADirectoryError as e:  # h5py 3.x
                raise IOError('Please specify a non-directory filepath for '
                              'ModelCheckpoint. Filepath used is an existing '
                              'directory: {}'.format(filepath))
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise e


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output_v3", help="output dir")
    parse.add_argument("--train_data", type=str, default="data/train_data_v2/train.txt", help="train data path")
    parse.add_argument("--dev_data", type=str, default="data/train_data_v2/test.txt", help="validation data path")
    parse.add_argument("--labels_name", type=str, default="data/train_data_v2/label.pickle",
                       help="labels names in student model")
    parse.add_argument("--labels_name_t", type=str, default="data/train_data/label.pickle",
                       help="labels name in teacher model")
    parse.add_argument("--teacher_model_path", type=str, help="already trained teacher model")
    parse.add_argument("--albert", type=str, default="BERT", choices=["BERT", "ALBERT"], help="bert type")
    parse.add_argument("--bert_model", type=str, default="E:\\resources\\albert_tiny_zh_google")
    parse.add_argument("--lr", type=float, default=1e-5)
    parse.add_argument("--batch_size", type=int, default=8, help="batch size")
    parse.add_argument("--max_length", type=int, default=128, help="max sequence length")
    parse.add_argument("--epochs", type=int, default=1, help="number of training epoch")
    parse.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal_loss", "asymmetric_loss"],
                       help="use bce for binary cross entropy loss and focal for focal loss")
    parse.add_argument("--temperature", type=int, default=3)
    # 参数配置
    args = parse.parse_args()
    # tokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_model, "vocab.txt"), pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=args.max_length)

    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)
    # 加载teacher模型
    label_encode_teacher = MultiLabelBinarizer()
    label_encode_student = MultiLabelBinarizer()
    word_dict = tokenizer.get_vocab()
    tokenizer.enable_truncation(max_length=args.max_length)
    processor = DataProcessorFunctions()
    # 设置类别标签
    label_t = processor.get_labels(args.labels_name_t)
    # label_t 和 label_s 相互独立
    label_s = processor.get_labels(args.labels_name)

    label_encode_teacher.fit([label_t])
    label_encode_student.fit([label_s])
    albert_config = os.path.join(args.bert_model, "config.json")
    teacher_config = AlbertConfig.from_pretrained(albert_config)
    teacher_config.num_labels = len(label_t)
    teacher_config.return_dict = True
    teacher_config.output_attentions = False
    teacher_config.output_hidden_states = False
    teacher_config.max_length = args.max_length
    model_teacher = TeacherTFAlbertForMLC.from_pretrained(args.teacher_model_path, config=teacher_config)
    # ###################冻结teacher模型的参数#####################
    model_teacher.trainable = False
    # #######################加载和构建student模型 ####################################
    student_config = AlbertConfig.from_pretrained(albert_config)
    student_config.teacher_num_labels = len(label_encode_teacher.classes_)
    student_config.num_labels = len(label_encode_student.classes_)
    student_config.return_dict = True
    student_config.output_attentions = False
    student_config.output_hidden_states = False
    student_config.max_length = args.max_length

    # ###################################student model初始化##################
    model_student = StudentTFAlbertForMLC.from_pretrained(args.bert_model,
                                                          config=student_config,
                                                          )
    # #######模型loss和metric：
    # if args.loss_type == "bce":
    #     model_student.compile(loss=model_student.compute_loss, optimizer=Adam(args.lr),
    #                           metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    # if args.loss_type == "focal_loss":
    #     model_student.compile(loss=model_student.compute_focal_loss, optimizer=Adam(args.lr),
    #                           metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    # #########################构建训练集和测试集 ####################

    train_examples = processor.get_train_examples(args.train_data)
    train_features = processor.convert_examples_to_features(train_examples,
                                                            tokenizer=tokenizer,
                                                            label_encode=label_encode_student)
    train_sequence = DataSequence(train_features,
                                  token_pad_id=tokenizer.token_to_id("[PAD]"),
                                  num_classes=len(label_encode_student.classes_),
                                  batch_size=args.batch_size)
    # 加载验证集数据
    dev_examples = processor.get_dev_examples(args.dev_data)
    dev_features = processor.convert_examples_to_features(dev_examples,
                                                          tokenizer=tokenizer,
                                                          label_encode=label_encode_student)
    dev_sequence = DataSequence(dev_features,
                                token_pad_id=tokenizer.token_to_id("[PAD]"),
                                num_classes=len(label_encode_student.classes_),
                                batch_size=args.batch_size)

    ######添加callback####
    call_backs = []
    log_dir = os.path.join(args.output_root, "logs")
    report = ClassificationReporter(validation_data=dev_sequence)
    call_backs.append(report)
    tensor_board = TensorBoard(log_dir=log_dir, write_graph=True, write_grads=True, histogram_freq=1)
    call_backs.append(tensor_board)
    model_dir = os.path.join(args.output_root, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint = StudentModelCheckpoint(os.path.join(model_dir, 'albert_ml.h5'), monitor='val_student_loss', verbose=1,
                                        save_best_only=True,
                                        mode='min',
                                        save_weights_only=True
                                        )
    call_backs.append(checkpoint)
    early_stop = EarlyStopping('val_student_loss', patience=5, mode='min', verbose=2, restore_best_weights=True)
    call_backs.append(early_stop)


    class DistillerModel(tf.keras.Model):
        def __init__(self, student, teacher):
            super(DistillerModel, self).__init__()
            self.teacher = teacher
            self.student = student

        def compile(
                self,
                optimizer,
                metrics,
                student_loss_fn,
                distillation_loss_fn,
                alpha=0.1,
                temperature=3,
        ):
            super(DistillerModel, self).compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature

        @tf.function
        def train_step(self, data):
            # Unpack data
            x, y = data
            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                student_predictions = self.student(x, training=True)
                logits, student_logits = student_predictions.logits
                student_loss = self.student_loss_fn(y, logits)
                st_post_dist = tf.expand_dims(tf.sigmoid(student_logits), axis=-1)
                st_neg_dist = 1 - st_post_dist
                st_dist = tf.concat([st_post_dist, st_neg_dist], axis=-1)
                th_post_dist = tf.expand_dims(tf.sigmoid(teacher_predictions.logits), axis=-1)
                th_neg_dist = 1 - th_post_dist
                th_dist = tf.concat([th_post_dist, th_neg_dist], axis=-1)
                distillation_loss = self.distillation_loss_fn(
                    th_dist, st_dist
                )
                loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`.
            self.compiled_metrics.update_state(y, student_predictions.logits[0])

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update(
                {"student_loss": student_loss, "distillation_loss": distillation_loss}
            )
            return results

        @tf.function
        def test_step(self, data):
            x, y = data
            y_prediction = self.student(x, training=False)
            student_loss = self.student_loss_fn(y, y_prediction.logits[0])
            self.compiled_metrics.update_state(y, y_prediction.logits[0])
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss})
            return results

        def call(self, inputs, training=None):
            out = self.student(inputs)
            return out.logits[0]


    # #############构建distill模型进行训练#########################################
    inputs_id = tf.keras.Input(shape=(None,), name="input_ids",
                               dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(None,), name="attention_mask",
                                    dtype=tf.int32)
    token_type_id = tf.keras.Input(shape=(None,), name="token_type_ids",
                                   dtype=tf.int32)

    # teacher_output = model_teacher({"input_ids": inputs_id, "token_type_ids": token_type_id})
    # student_output = model_student({"input_ids": inputs_id, "token_type_ids": token_type_id})
    # logits, student_logits = tf.keras.layers.Lambda(lambda x: x.logits)(student_output)
    #
    # student_positive_distribute = tf.keras.activations.sigmoid(student_logits)
    # student_positive_distribute = tf.keras.backend.expand_dims(student_positive_distribute, axis=-1)
    # student_negative_distribute = tf.keras.layers.Lambda(lambda x: 1 - x)(student_positive_distribute)
    # student_distribute = tf.keras.backend.concatenate([student_positive_distribute, student_negative_distribute],
    #                                                   axis=-1)
    # teacher_positive_distribute = tf.keras.activations.sigmoid(teacher_output.logits)
    # teacher_positive_distribute = tf.keras.backend.expand_dims(teacher_positive_distribute, axis=-1)
    # teacher_negative_distribute = tf.keras.layers.Lambda(lambda x: 1 - x)(teacher_positive_distribute)
    # teacher_distribute = tf.keras.backend.concatenate([teacher_positive_distribute, teacher_negative_distribute],
    #                                                   axis=-1)
    #
    # # ##############loss计算############################################################
    # teacher_loss_fn = tf.keras.losses.KLDivergence(
    #     reduction=tf.keras.losses.Reduction.SUM
    # )
    #
    # # loss_fn_teacher = tf.keras.losses.BinaryCrossentropy(
    # #     from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    # # )
    # kl_loss = tf.keras.layers.Lambda(
    #     lambda x: teacher_loss_fn(x[0], x[1]), name="kl_loss_computer")(
    #     [teacher_distribute, student_distribute])
    #
    # model = tf.keras.Model(inputs=[inputs_id, token_type_id], outputs=[logits],
    #                        name="distiller")
    #
    # model.add_loss(kl_loss)
    # model.add_metric(tf.reduce_mean(kl_loss), name="kl_loss")
    # if args.loss_type == "bce":
    #     model.compile(
    #         loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM),
    #         optimizer=Adam(args.lr),
    #         metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    # elif args.loss_type == "focal_loss":
    #     model.compile(
    #         loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM),
    #         optimizer=Adam(args.lr),
    #         metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    #     )
    # model_student.summary()

    model = DistillerModel(student=model_student, teacher=model_teacher)

    if args.loss_type == "bce":
        model.compile(optimizer=Adam(args.lr),
                      student_loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                                         reduction=tf.keras.losses.Reduction.SUM),
                      distillation_loss_fn=tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                      )

    elif args.loss_type == "focal_loss":

        model.compile(optimizer=Adam(args.lr),
                      student_loss_fn=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False,
                                                                          reduction=tf.keras.losses.Reduction.SUM),
                      distillation_loss_fn=tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                      )
    elif args.loss_type == "asymmetric_loss":

        model.compile(optimizer=Adam(args.lr),
                      student_loss_fn=AsymmetricLoss(from_logits=False,
                                                     reduction=tf.keras.losses.Reduction.SUM),
                      distillation_loss_fn=tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                      )

    model.fit(train_sequence,
              validation_data=dev_sequence,
              steps_per_epoch=len(train_sequence),
              epochs=args.epochs,
              shuffle=True,
              # use_multiprocessing=False,
              # train_sequence注意使用可序列化的对象
              callbacks=call_backs,
              # 设置分类权重
              )
    model.summary()
    # plot_model(model, show_shapes=True, show_layer_names=True)
