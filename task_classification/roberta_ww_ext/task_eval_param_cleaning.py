# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_eval_param_cleaning.py
@time: 2022/3/10 13:52
"""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from keras.utils.data_utils import Sequence
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from transformers.models.bert import BertConfig, BertTokenizer

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

            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             text_c=text_c,
                             text_d=text_d,
                             label=None
                             ))
        return examples

    def get_test_examples(self, data_path):
        pass

    @staticmethod
    def get_labels(path=None):
        return ["positive", "negative"]

    @staticmethod
    def convert_single_example(example: InputExample,
                               tokenizer: BertTokenizer,
                               label_encode: LabelEncoder = None,
                               max_length: int = 128):
        sentence = "[SEP]".join([example.text_a, example.text_b, example.text_c, example.text_d])
        encoder = tokenizer(sentence, max_length=max_length,truncation=True)
        input_ids = encoder.input_ids
        segment_ids = encoder.token_type_ids
        input_mask = encoder.attention_mask
        feature = InputFeaturesBase(
            guid=example.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=None,
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
                "token_type_ids": batch_segment_ids}


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--output_root", type=str, default="output", help="output dir")
    parse.add_argument("--infer_data", type=str, default="data/param_cleaning/train.txt", help="train data path")
    parse.add_argument("--bert_model", type=str, default="E:\\Resources\\chinese-roberta-wwm-ext")
    parse.add_argument("--saved_model", type=str, default="output/model/roberta.h5", help="saved model")
    parse.add_argument("--batch_size", type=int, default=8, help="batch size")
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
    bert_config = os.path.join(args.bert_model, "config.json")
    config = BertConfig.from_pretrained(bert_config)
    # 更新基本参数
    config.num_labels = len(label_encode.classes_)
    config.return_dict = True
    config.output_attentions = False
    config.output_hidden_states = False
    config.max_length = args.max_length


    #######################################################################
    # train_examples = processor.get_train_examples(args.infer_data)
    # train_features = processor.convert_examples_to_features(train_examples,
    #                                                         tokenizer=tokenizer,
    #                                                         max_length=args.max_length,
    #                                                         label_encode=label_encode)

    # strategy = tf.distribute.MirroredStrategy()
    # logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # logger.info("loading saved model....")
    # with strategy.scope():
    #     model = TFBertForSequenceClassification.from_pretrained(args.saved_model,
    #                                                             config=config
    #                                                             )
    #     logger.info("loading model success")
    #
    # example_ids_map = {}
    # for example in train_examples:
    # example_ids_map[example.guid] = "[SEP]".join([example.text_a, example.text_b, example.text_c, example.text_d])
    #  单例预测
    # feature_ids_map = {}
    # for feature in train_features:
    #     out = model({"input_ids": np.array([feature.input_ids]), "token_type_ids": np.array([feature.segment_ids])})
    #     pro = tf.nn.softmax(out.logits)
    #     label = np.argmax(pro, axis=-1)
    #     feature_ids_map[feature.guid] = [pro, label]
    #
    # for k, v in example_ids_map.items():
    #     print(v, feature_ids_map[k])

    # # 全量预测
    # train_ds = DataSequence(train_features,
    #                         token_pad_id=tokenizer.pad_token_id,
    #                         num_classes=len(label_encode.classes_),
    #                         batch_size=args.batch_size)
    # for d in train_ds:
    #     out = model(d)
    #     pro = tf.nn.softmax(out.logits)
    #     label_index = np.argmax(pro, axis=-1)
    #     label = label_encode.inverse_transform(label_index)
    #     label_pro = np.take_along_axis(pro.numpy(), np.expand_dims(label_index, axis=-1), axis=1)

    # 尝试适用datasets
    # def encode(examples):
    #     return tokenizer(examples['gds_nm'], examples['l4_gds_desc'], truncation=True, padding='max_length')
    #
    #
    # data = load_dataset(path="csv",
    #                     data_files=["data/param_cleaning/train.txt"],
    #                     column_names=["l4_gds_desc", "gds_nm", "attr", "value", "label"],
    #                     delimiter="\t",
    #                     encoding="utf-8",
    #                     )
    #
    # data = data.map(encode, batched=True,batch_size=2)
    # for d in data["train"]:
    #     print(d)
