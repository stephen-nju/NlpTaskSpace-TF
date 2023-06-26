# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_eval_student.py
@time: 2022/1/21 16:06
"""
import argparse
import csv
import json
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import Sequence
from tokenizers.implementations import BertWordPieceTokenizer
from tqdm import tqdm
from transformers.file_utils import ModelOutput
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
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 guid,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
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
        lines = self._read_tsv(data_path)
        examples = []
        for (i, line) in enumerate(lines):
            # if i > 20:
            #     break
            guid = "infer-%d" % (i)
            text_a = convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=None))
        return examples

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
                lines.append([line[0], None])
            return lines

    @staticmethod
    def convert_single_example(example, tokenizer):

        encoder = tokenizer.encode(sequence=example.text_a)
        input_ids = encoder.ids
        segment_ids = encoder.type_ids
        input_mask = encoder.attention_mask
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=None,
            guid=example.guid,
            is_real_example=True)
        return feature

    def convert_examples_to_features(self, examples, tokenizer):
        features = []
        # 这里可以采用多线程等方式提高预处理速度
        for example in examples:
            feature = self.convert_single_example(example=example, tokenizer=tokenizer)
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
        batch_token_ids, batch_segment_ids, batch_attention_mask_ids, batch_guids = [], [], [], []
        for feature in features:
            batch_token_ids.append(feature.input_ids)
            batch_segment_ids.append(feature.segment_ids)
            batch_attention_mask_ids.append(feature.input_mask)
            batch_guids.append(feature.guid)
        batch_token_ids = sequence_padding(batch_token_ids, value=self.token_pad_id)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        batch_attention_mask_ids = sequence_padding(batch_attention_mask_ids)
        return [batch_token_ids, batch_attention_mask_ids, batch_segment_ids], batch_guids


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


class TFAlbertDistillLoss:
    """
    Loss function suitable for sequence classification.
    """

    def compute_loss(self, labels, logits):
        print("label_shape", labels.shape)
        # print("logits shape", logits[0].shape)
        # print("logits shape2", logits[1].shape)
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        # teacher_loss_fn = tf.keras.losses.BinaryCrossentropy(
        #     from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        # )
        return loss_fn(labels, logits)

    def compute_teacher_loss(self, teacher, student):
        teacher_loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        return teacher_loss_fn(teacher, student)

    def compute_focal_loss(self, labels, logits):
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
        teacher_loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        return loss_fn(labels, logits)


@dataclass
class StudentTFAlbertOutput(ModelOutput):
    loss: Optional[tf.Tensor] = None
    logits: Tuple[tf.Tensor, tf.Tensor] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


class StudentTFAlbertForMLC(TFAlbertPreTrainedModel, TFAlbertDistillLoss):
    _keys_to_ignore_on_load_unexpected = [r"predictions"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.teacher_num_labels = config.teacher_num_labels
        self.albert = TFAlbertMainLayer(config, name="albert")
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)
        self.student_classifier = tf.keras.layers.Dense(units=self.teacher_num_labels,
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

        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"],
                                                                       logits=out_logits)
        # loss = None
        # if inputs["labels"] is not None:
        #     loss = self.compute_loss(labels=inputs["labels"], logits=out_logits)

        if not inputs["return_dict"]:
            output = (out_logits, student_logits) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        return StudentTFAlbertOutput(
            loss=loss,
            logits=(out_logits, student_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification.serving_output
    def serving_output(self, output: StudentTFAlbertOutput) -> StudentTFAlbertOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return StudentTFAlbertOutput(logits=output.logits,
                                     hidden_states=hs,
                                     attentions=attns)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--saved_model", type=str, default="output_v3/model/albert_ml.h5", help="saved model path")
    parse.add_argument("--infer_data", type=str, default="D:\\Users\\Desktop\\tt.txt",
                       help="validation data path")

    parse.add_argument("--bert_model", type=str, default="E:\\resources\\albert_tiny_zh_google")
    parse.add_argument("--labels_name", type=str, default="data/train_data_v2/label.pickle",
                       help="labels names in student model")
    parse.add_argument("--labels_name_t", type=str, default="data/train_data/label.pickle",
                       help="labels name in teacher model")
    parse.add_argument("--batch_size", type=int, default=8, help="batch size")
    parse.add_argument("--max_length", type=int, default=128, help="max sequence length")

    # 参数配置
    args = parse.parse_args()
    # tokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_model, "vocab.txt"), pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=args.max_length)

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

    # #######################加载和构建student模型 ####################################
    student_config = AlbertConfig.from_pretrained(albert_config)
    student_config.teacher_num_labels = len(label_encode_teacher.classes_)
    student_config.num_labels = len(label_encode_student.classes_)
    student_config.return_dict = True
    student_config.output_attentions = False
    student_config.output_hidden_states = False
    student_config.max_length = args.max_length

    # ###################################加载模型#########################################
    model_student = StudentTFAlbertForMLC.from_pretrained(args.saved_model,
                                                          config=student_config,
                                                          )
    infer_examples = processor.get_test_examples(args.infer_data)
    infer_features = processor.convert_examples_to_features(infer_examples,
                                                            tokenizer=tokenizer)
    infer_sequence = DataSequence(infer_features,
                                  token_pad_id=tokenizer.token_to_id("[PAD]"),
                                  num_classes=len(label_encode_student.classes_),
                                  batch_size=args.batch_size)

    guid_result_dict = defaultdict()
    guid_result_dict_new = defaultdict()
    for data, guids in tqdm(infer_sequence):
        out = model_student(data)
        new, old = out.logits
        sigmoid_data = 1 / (1 + np.exp(-old.numpy()))
        infer_out_old = np.where(sigmoid_data > 0.5, 1, 0)
        infer_out_new = np.where(new > 0.5, 1, 0)
        infer_label_old = label_encode_teacher.inverse_transform(infer_out_old)
        infer_label_new = label_encode_student.inverse_transform(infer_out_new)
        for label, label_new, guid in zip(infer_label_old, infer_label_new, guids):
            guid_result_dict_new.setdefault(guid, []).append(label_new)
            guid_result_dict.setdefault(guid, []).append(label)

    guid_examples_dict = defaultdict()
    for example in infer_examples:
        guid_examples_dict.setdefault(example.guid, []).append(example.text_a)

    out_result = []
    for guid, label in guid_result_dict.items():
        if guid in guid_examples_dict:
            out_result.append((guid_examples_dict[guid][0], json.dumps(label, ensure_ascii=False)))
    out_result_new = []
    for guid, label in guid_result_dict_new.items():
        if guid in guid_examples_dict:
            out_result_new.append((guid_examples_dict[guid][0], json.dumps(label, ensure_ascii=False)))

    with open("D:/Users/Desktop/out_student.txt", "w", encoding="utf-8") as g:
        for o in out_result:
            g.write("\t".join(o) + "\n")

    with open("D:/Users/Desktop/out_student_new.txt", "w", encoding="utf-8") as g:
        for o in out_result_new:
            g.write("\t".join(o) + "\n")
