# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_consolidation_train.py
@time: 2022/2/10 10:48
"""

import argparse
import csv
import json
import os
import pickle
import random
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
# from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, TensorBoard, EarlyStopping
from tokenizers.implementations import BertWordPieceTokenizer
from transformers.file_utils import ModelOutput
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.modeling_tf_utils import get_initializer, TFModelInputType, input_processing
from transformers.models.albert import AlbertConfig, TFAlbertPreTrainedModel, TFAlbertMainLayer

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
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.guid = guid
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
        lines = self._read_tsv_distill(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 20:
            #    break
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
    def _read_tsv_distill(cls, input_file, quotechar=None):
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append([line[0], None])
            return lines

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
    def convert_single_example(example, tokenizer, label_encode, training=True):
        encoder = tokenizer.encode(sequence=example.text_a)
        input_ids = encoder.ids
        segment_ids = encoder.type_ids
        input_mask = encoder.attention_mask
        if training:
            label_id = label_encode.transform([example.label])
            feature = InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                is_real_example=True)
        else:
            feature = InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=[[None]],
                is_real_example=True)

        return feature

    def convert_examples_to_features(self, examples, tokenizer, label_encode=None, training=True):
        features = []
        # 这里可以采用多线程等方式提高预处理速度
        for example in examples:
            feature = self.convert_single_example(example=example,
                                                  tokenizer=tokenizer,
                                                  label_encode=label_encode,
                                                  training=training)
            features.append(feature)

        return features


class DataSequence(Sequence):
    def __init__(self, features, token_pad_id, batch_size, training=True):
        self.batch_size = batch_size
        self.features = features
        self.token_pad_id = token_pad_id
        self.training = training

    def __len__(self):
        # 删除batch_size 不足的数据
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, index):
        data = self.features[index * self.batch_size:(index + 1) * self.batch_size]
        # print(index, len(data))
        return self.feature_batch_transform(data)

    def feature_batch_transform(self, features):
        batch_token_ids, batch_segment_ids, batch_attention_mask_ids, batch_labels, batch_guids = [], [], [], [], []
        for feature in features:
            batch_token_ids.append(feature.input_ids)
            batch_segment_ids.append(feature.segment_ids)
            batch_attention_mask_ids.append(feature.input_mask)
            batch_labels.append(feature.label_id)
            batch_guids.append(feature.guid)

        batch_token_ids = sequence_padding(batch_token_ids, value=self.token_pad_id)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        batch_attention_mask_ids = sequence_padding(batch_attention_mask_ids)
        if self.training:
            return [batch_token_ids, batch_segment_ids], np.squeeze(
                np.array(batch_labels, dtype=float), axis=1)
        else:
            return [batch_token_ids, batch_segment_ids], batch_guids


class TeacherTFAlbertForMLC(TFAlbertPreTrainedModel, ABC):
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

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return output

        return TFSequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class ConsolidatedTFAlbertOutput(ModelOutput):
    logits: Tuple[tf.Tensor, tf.Tensor] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


class ConsolidatedTFAlbertForMLC(TFAlbertPreTrainedModel, ABC):
    _keys_to_ignore_on_load_unexpected = [r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.old_num_labels = config.old_num_labels
        self.incremental_num_labels = config.incremental_num_labels
        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.old_classifier = tf.keras.layers.Dense(units=self.old_num_labels,
                                                    kernel_initializer=get_initializer(config.initializer_range),
                                                    name="old_model_classifier")
        self.incremental_classifier = tf.keras.layers.Dense(units=self.incremental_num_labels,
                                                            kernel_initializer=get_initializer(
                                                                config.initializer_range),
                                                            name="incremental_classifier",
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
    ) -> Union[ConsolidatedTFAlbertOutput, Tuple[tf.Tensor]]:
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

        old_logits = self.old_classifier(inputs=pooled_output)
        incremental_logits = self.incremental_classifier(inputs=pooled_output)

        if not inputs["return_dict"]:
            output = (old_logits, incremental_logits) + outputs[2:]
            return output

        return ConsolidatedTFAlbertOutput(
            logits=(old_logits, incremental_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--saved_model", type=str, default="output_consolidated/model/albert_distill",
                       help="saved model")
    parse.add_argument("--train_data", type=str, default="data/train_data_v2/train.txt", help="train data path")
    parse.add_argument("--dev_data", type=str, default="data/train_data_v2/test.txt", help="validation data path")
    parse.add_argument("--labels_name_old", type=str, default="data/train_data/label.pickle",
                       help="labels names in student model")
    parse.add_argument("--labels_name_incremental", type=str, default="data/train_data_v2/label.pickle",
                       help="labels name in teacher model")
    parse.add_argument("--old_teacher_model_path", type=str, help="already trained teacher model")
    parse.add_argument("--incremental_teacher_model_path", type=str, help="already trained teacher model")
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

    # 加载teacher模型
    label_encode_old = MultiLabelBinarizer()
    label_encode_incremental = MultiLabelBinarizer()
    word_dict = tokenizer.get_vocab()
    tokenizer.enable_truncation(max_length=args.max_length)
    processor = DataProcessorFunctions()
    # 设置类别标签
    label_t = processor.get_labels(args.labels_name_old)
    # label_t 和 label_s 相互独立
    label_s = processor.get_labels(args.labels_name_incremental)

    label_encode_old.fit([label_t])
    label_encode_incremental.fit([label_s])
    albert_config = os.path.join(args.bert_model, "config.json")
    # #########################teacher_model_old############################################
    old_teacher_config = AlbertConfig.from_pretrained(albert_config)
    old_teacher_config.num_labels = len(label_t)
    old_teacher_config.return_dict = True
    old_teacher_config.output_attentions = False
    old_teacher_config.output_hidden_states = False
    old_teacher_config.max_length = args.max_length
    model_teacher_old = TeacherTFAlbertForMLC.from_pretrained(args.old_teacher_model_path,
                                                              config=old_teacher_config)
    # ###################冻结teacher模型的参数#####################
    model_teacher_old.trainable = False

    # ###################teacher_model_incremental#################################################
    incremental_teacher_config = AlbertConfig.from_pretrained(albert_config)
    incremental_teacher_config.num_labels = len(label_s)
    incremental_teacher_config.return_dict = True
    incremental_teacher_config.output_attentions = False
    incremental_teacher_config.output_hidden_states = False
    incremental_teacher_config.max_length = args.max_length
    model_teacher_incremental = TeacherTFAlbertForMLC.from_pretrained(args.incremental_teacher_model_path,
                                                                      config=incremental_teacher_config)
    model_teacher_incremental.trainable = False

    # #######################加载和构建student模型 ####################################
    consolidation_model_config = AlbertConfig.from_pretrained(albert_config)
    consolidation_model_config.old_num_labels = len(label_encode_old.classes_)
    consolidation_model_config.incremental_num_labels = len(label_encode_incremental.classes_)
    consolidation_model_config.return_dict = True
    consolidation_model_config.output_attentions = False
    consolidation_model_config.output_hidden_states = False
    consolidation_model_config.max_length = args.max_length

    # ###################################student model初始化##################
    model_consolidation = ConsolidatedTFAlbertForMLC.from_pretrained(args.bert_model,
                                                                     config=consolidation_model_config,
                                                                     )
    #
    # train_examples = processor.get_train_examples(args.train_data)
    # train_features = processor.convert_examples_to_features(train_examples,
    #                                                         tokenizer=tokenizer,
    #                                                         label_encode=None,
    #                                                         training=False,
    #                                                         # ##training参数控制label标签是否为None
    #                                                         )
    #
    # # #######training 参数控制是否传入label###########################
    # train_sequence = DataSequence(train_features,
    #                               token_pad_id=tokenizer.token_to_id("[PAD]"),
    #                               batch_size=args.batch_size,
    #                               training=True
    #                               )

    # 加载验证集数据
    dev_examples = processor.get_dev_examples(args.dev_data)
    dev_features = processor.convert_examples_to_features(dev_examples,
                                                          tokenizer=tokenizer,
                                                          label_encode=label_encode_incremental,
                                                          training=True
                                                          )
    dev_sequence = DataSequence(dev_features,
                                token_pad_id=tokenizer.token_to_id("[PAD]"),
                                batch_size=args.batch_size
                                )


    class DistillerModel(tf.keras.Model):
        def __init__(self, student, old_teacher, incremental_teacher):
            super(DistillerModel, self).__init__()
            self.old_teacher = old_teacher
            self.incremental_teacher = incremental_teacher
            self.student = student

        def compile(
                self,
                optimizer,
                distillation_loss_fn_old,
                distillation_loss_fn_incremental,
                alpha=0.5,
                temperature=3,
                metrics=None,
        ):
            super(DistillerModel, self).compile(optimizer=optimizer, metrics=metrics)
            self.distillation_loss_fn_old = distillation_loss_fn_old
            self.distillation_loss_fn_incremental = distillation_loss_fn_incremental
            self.alpha = alpha
            self.temperature = temperature

        @tf.function
        def train_step(self, data):
            # Unpack data
            x = data
            old_teacher_predictions = self.old_teacher(x, training=False)
            incremental_teacher_predictions = self.incremental_teacher(x, training=False)
            with tf.GradientTape() as tape:
                student_predictions = self.student(x, training=True)
                old_logits, incremental_logits = student_predictions.logits
                st_post_dist = tf.expand_dims(tf.sigmoid(old_logits), axis=-1)
                st_neg_dist = 1 - st_post_dist
                st_dist = tf.concat([st_post_dist, st_neg_dist], axis=-1)
                th_post_dist = tf.expand_dims(tf.sigmoid(old_teacher_predictions.logits), axis=-1)
                th_neg_dist = 1 - th_post_dist
                th_dist = tf.concat([th_post_dist, th_neg_dist], axis=-1)
                old_loss = self.distillation_loss_fn_old(
                    th_dist, st_dist
                )

                in_st_post_dist = tf.expand_dims(tf.sigmoid(incremental_logits), axis=-1)
                in_st_neg_dist = 1 - in_st_post_dist
                in_st_dist = tf.concat([in_st_post_dist, in_st_neg_dist], axis=-1)
                in_th_post_dist = tf.expand_dims(tf.sigmoid(incremental_teacher_predictions.logits), axis=-1)
                in_th_neg_dist = 1 - in_th_post_dist
                in_th_dist = tf.concat([in_th_post_dist, in_th_neg_dist], axis=-1)
                incremental_loss = self.distillation_loss_fn_incremental(
                    in_th_dist, in_st_dist
                )
                loss = self.alpha * old_loss + (1 - self.alpha) * incremental_loss

            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`.
            # self.compiled_metrics.update_state(y, student_predictions.logits[0])

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update(
                {"loss": loss}
            )
            return results

        # @tf.function
        # def test_step(self, data):
        #     x, y = data
        #     y_prediction = self.student(x, training=False)
        #     old_teacher_predictions = self.old_teacher(x, training=False)
        #     incremental_teacher_predictions = self.incremental_teacher(x, training=False)
        #     old_loss = self.distillation_loss_fn_old(old_teacher_predictions, y_prediction.logits[0])
        #     incremental_loss = self.distillation_loss_fn_incremental(incremental_teacher_predictions,
        #                                                              y_prediction.logits[1])
        #
        #     # self.compiled_metrics.update_state(y, y_prediction.logits[0])
        #     results = {m.name: m.result() for m in self.metrics}
        #     loss = self.alpha * old_loss + (1 - self.alpha) * incremental_loss
        #     results.update({"loss": loss})
        #     return results

        def call(self, inputs, training=None):
            out = self.student(inputs)
            return out.logits


    # #############构建distill模型进行训练#########################################
    # inputs_id = tf.keras.Input(shape=(None,), name="input_ids",
    #                            dtype=tf.int32)
    # attention_mask = tf.keras.Input(shape=(None,), name="attention_mask",
    #                                 dtype=tf.int32)
    # token_type_id = tf.keras.Input(shape=(None,), name="token_type_ids",
    #                                dtype=tf.int32)

    model = DistillerModel(student=model_consolidation,
                           old_teacher=model_teacher_old,
                           incremental_teacher=model_teacher_incremental)

    model.compile(optimizer=Adam(args.lr),
                  distillation_loss_fn_old=tf.keras.losses.KLDivergence(
                      reduction=tf.keras.losses.Reduction.SUM),
                  distillation_loss_fn_incremental=tf.keras.losses.KLDivergence(
                      reduction=tf.keras.losses.Reduction.SUM),
                  )

    # model.fit(train_sequence,
    #           # validation_data=dev_sequence,
    #           steps_per_epoch=len(train_sequence),
    #           epochs=args.epochs,
    #           shuffle=True,
    #           # use_multiprocessing=False,
    #           # train_sequence注意使用可序列化的对象
    #           # 设置分类权重
    #           )

    # model.summary()
    ######
    print("############laoding model weights##############")
    model.load_weights(args.saved_model)
    print("#############laoding model weights success#####")

    old_model_res_guid_dict = defaultdict()
    incremental_model_res_guid__dict = defaultdict()
    for (data, guids) in dev_sequence:
        out = model.student(data)
        old, incremental_logits = out.logits
        sigmoid_data = 1 / (1 + np.exp(-old.numpy()))
        infer_out_old = np.where(sigmoid_data > 0.5, 1, 0)
        infer_out_new = np.where((1 / (1 + np.exp(-incremental_logits.numpy()))) > 0.5, 1, 0)
        infer_label_old = label_encode_old.inverse_transform(infer_out_old)
        infer_label_new = label_encode_incremental.inverse_transform(infer_out_new)
        for label, label_new, guid in zip(infer_label_old, infer_label_new, guids):
            old_model_res_guid_dict.setdefault(guid, []).append(label_new)
            incremental_model_res_guid__dict.setdefault(guid, []).append(label)

    guid_examples_dict = defaultdict()
    for example in dev_examples:
        guid_examples_dict.setdefault(example.guid, []).append(example.text_a)

    old_model_out_res = []
    for guid, label in incremental_model_res_guid__dict.items():
        if guid in guid_examples_dict:
            old_model_out_res.append((guid_examples_dict[guid][0], json.dumps(label, ensure_ascii=False)))
    incremental_model_out_res = []
    for guid, label in old_model_res_guid_dict.items():
        if guid in guid_examples_dict:
            incremental_model_out_res.append((guid_examples_dict[guid][0], json.dumps(label, ensure_ascii=False)))

    with open("old_output.txt", "w", encoding="utf-8") as g:
        for o in old_model_out_res:
            g.write("\t".join(o) + "\n")

    with open("incremental_output.txt", "w", encoding="utf-8") as g:
        for o in incremental_model_out_res:
            g.write("\t".join(o) + "\n")
