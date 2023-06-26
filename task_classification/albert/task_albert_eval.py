# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_albert_ml_classification.py
@time: 2021/9/10 16:05
"""
import argparse
import csv
import json
import os
import pickle
import random
from abc import ABCMeta
from collections import defaultdict
from typing import Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import Sequence
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import TFAlbertMainLayer
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.modeling_tf_utils import get_initializer, TFModelInputType, input_processing
from transformers.models.albert import AlbertConfig, TFAlbertPreTrainedModel

from dependence.snippets import convert_to_unicode, sequence_padding


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
                 guid,
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
        batch_token_ids, batch_segment_ids, batch_guids = [], [], []
        for feature in features:
            batch_token_ids.append(feature.input_ids)
            batch_segment_ids.append(feature.segment_ids)
            batch_guids.append(feature.guid)
        batch_token_ids = sequence_padding(batch_token_ids, value=self.token_pad_id)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        return {"input_ids": batch_token_ids,
                "token_type_ids": batch_segment_ids}, batch_guids


class TFMultiLabelClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """

    def compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )

        return loss_fn(labels, logits)


class TFAlbertForMultiLabelClassification(TFAlbertPreTrainedModel, metaclass=ABCMeta):
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
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return output

        return TFSequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # if not inputs["return_dict"]:
        #     output = (logits,) + outputs[2:]
        #
        #     return ((loss,) + output) if loss is not None else output
        #
        # return TFSequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    # # Copied from transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification.serving_output
    # def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
    #     hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
    #     attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
    #
    #     return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output", help="output dir")
    parse.add_argument("--infer_data", type=str, default="D:\\Users\\Desktop\\dd.txt",
                       help="validation data path")

    parse.add_argument("--labels_name", type=str, default="data/train_data/label.pickle",
                       help="labels names in data")
    parse.add_argument("--bert_model", type=str, default="E:\\resources\\albert_tiny_zh_google")
    parse.add_argument("--saved_model", type=str, default=None, help="saved model path")
    parse.add_argument("--lr", type=float, default=1e-5)
    parse.add_argument("--batch_size", type=int, default=8, help="batch size")
    parse.add_argument("--max-length", type=int, default=128, help="max sequence length")
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
    labels = processor.get_labels(args.labels_name)
    label_encode.fit([labels])
    # 二维列表
    infer_examples = processor.get_test_examples(args.infer_data)
    infer_features = processor.convert_examples_to_features(infer_examples,
                                                            tokenizer=tokenizer)
    infer_sequence = DataSequence(infer_features,
                                  token_pad_id=tokenizer.token_to_id("[PAD]"),
                                  num_classes=len(label_encode.classes_),
                                  batch_size=args.batch_size)

    albert_config = os.path.join(args.bert_model, "config.json")
    config = AlbertConfig.from_pretrained(albert_config)
    # 更新基本参数
    config.num_labels = len(label_encode.classes_)
    config.return_dict = True
    config.output_attentions = False
    config.output_hidden_states = False
    config.max_length = args.max_length
    model = TFAlbertForMultiLabelClassification.from_pretrained(args.saved_model, config=config)

    guid_result_dict = defaultdict()
    for data, guids in infer_sequence:
        out = model(data)
        out_prob = out.logits.numpy()
        out = np.where(out_prob > 0.5, 1, 0)
        probs = [np.compress(o, p).round(3).tolist() for o, p in zip(out, out_prob)]
        infer_out = label_encode.inverse_transform(out)
        for label, prob, guid in zip(infer_out, probs, guids):
            guid_result_dict.setdefault(guid, []).append([label, prob])

    guid_examples_dict = defaultdict()
    for example in infer_examples:
        guid_examples_dict.setdefault(example.guid, []).append(example.text_a)

    out_result = []
    for guid, label in guid_result_dict.items():
        if guid in guid_examples_dict:
            out_result.append((guid_examples_dict[guid][0], json.dumps(label, ensure_ascii=False)))

    with open("D:\\Users\\Desktop\\no_match_out_l.txt", "w", encoding="utf-8") as g:
        for o in out_result:
            g.write("\t".join(o) + "\n")
    print("end processing")
