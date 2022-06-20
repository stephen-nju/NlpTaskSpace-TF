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
import json
import os
import random

import numpy as np
import tensorflow as tf
from keras.utils.data_utils import Sequence
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.distribute.values import PerReplica
from tensorflow.python.keras.engine import data_adapter
from transformers.models.bert.modeling_tf_bert import TFBertForSequenceClassification
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
        encoder = tokenizer(sentence, max_length=max_length, truncation=True)
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
        batch_token_ids, batch_segment_ids, batch_labels, guids = [], [], [], []
        for feature in features:
            guids.append(feature.guid)
            batch_token_ids.append(feature.input_ids)
            batch_segment_ids.append(feature.segment_ids)
            batch_labels.append(feature.label_id)

        batch_token_ids = sequence_padding(batch_token_ids, value=self.token_pad_id)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        return {"input_ids": batch_token_ids,
                "token_type_ids": batch_segment_ids}, guids


class TFBertAdversarialTrain(TFBertForSequenceClassification):

    def compile(
            self,
            optimizer="rmsprop",
            loss="passthrough",
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            **kwargs
    ):
        super(TFBertAdversarialTrain, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs
        )

        # 1. 需要利用train_variables实例这种变量
        # 2. 采用延迟添加的方式
        # 3.如果在train_step 中间添加会与tf.function造成冲突
        # 4. tensorflow2.3 版本一下采用此种方案
        """Using tf.Variable() should be avoided inside the training loop, since it will produce errors when trying to execute the code as a graph.a
        If you use tf.Variable() inside your training function and then decorate it with "@tf.function" or apply "tf.function(my_train_fcn)" to obtain a graph function 
        (i.e. for improved performance), the execution will rise an error
        . This happens because the tracing of the tf.Variable function results in a different behaviour than the observed in eager execution (re-utilization or creation, respectively). You can find more info on this in the"""
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in self.trainable_variables]

    @tf.function
    def train_step(self, data):
        """
    计算在embedding上的gradient
    计算扰动 在embedding上加上扰动
    重新计算loss和gradient
    删除embedding上的扰动，并更新参数
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # These next two lines differ from the base method - they avoid issues when the labels are in
        # the input dict (and loss is computed internally)
        if y is None and "labels" in x:
            y = x["labels"]  # Stops confusion with metric computations
        # Run forward pass.
        # gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in
        #                          self.trainable_variables]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(grads[i])

        # embedding = model.trainable_variables[0]
        # embedding_gradients = tape.gradient(loss, [self.trainable_variables[0]])[0]
        # embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
        embedding_gradients = self.gradient_accumulation[0]
        delta = 0.2 * embedding_gradients / (tf.math.sqrt(tf.reduce_sum(embedding_gradients ** 2)) + 1e-8)  # 计算扰动
        self.trainable_variables[0].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            loss_adv = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # 累加对抗梯度
        gradients_adv = tape2.gradient(loss_adv, model.trainable_variables)

        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients_adv[i])

        self.trainable_variables[0].assign_sub(delta)
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        # These next two lines are also not in the base method - they correct the displayed metrics
        # when we're using a dummy loss, to avoid a bogus "loss_loss" value being shown.
        # 清空累计梯度
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
        if "loss" in return_metrics and "loss_loss" in return_metrics:
            del return_metrics["loss_loss"]
        return return_metrics


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--infer_data", type=str, default="data/param_cleaning/train.txt", help="train data path")
    parse.add_argument("--bert_model", type=str, default="E:\\Resources\\chinese-roberta-wwm-ext")
    parse.add_argument("--saved_model", type=str, default="output/model/roberta.h5", help="saved model")
    parse.add_argument("--batch_size", type=int, default=8, help="batch size")
    parse.add_argument("--max_length", type=int, default=128, help="max sequence length")
    parse.add_argument("--input_file", type=str, default="data/param_cleaning/train.txt",
                       help="input infer file path")
    parse.add_argument("--output_file", default="data/param_cleaning/out.txt", type=str,
                       help="output file path")
    # 参数配置
    args = parse.parse_args()
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

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
    examples = processor.get_train_examples(args.input_file)
    features = processor.convert_examples_to_features(examples,
                                                      tokenizer=tokenizer,
                                                      max_length=args.max_length,
                                                      label_encode=label_encode)

    strategy = tf.distribute.MirroredStrategy()

    logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    logger.info("loading saved model....")
    with strategy.scope():
        model = TFBertAdversarialTrain.from_pretrained(args.saved_model,
                                                       config=config
                                                       )
        logger.info("loading model success")

    # example_ids_map = {}
    # for example in examples:
    #     example_ids_map[example.guid] = "[SEP]".join([example.text_a, example.text_b, example.text_c, example.text_d])
    # 单例预测
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
    # keras sequence 预测
    # train_ds = DataSequence(features,
    #                         token_pad_id=tokenizer.pad_token_id,
    #                         num_classes=len(label_encode.classes_),
    #                         batch_size=args.batch_size)
    #
    # feature_ids_map = {}
    # for data_feature, guids in train_ds:
    #
    #
    #     distribute_value = strategy.experimental_distribute_values_from_function(value_fn)
    #     distribute_out = strategy.run(model, args=(distribute_value,))
    #     # out=strategy.reduce(distribute_out)
    #
    #     out = model(data_feature)
    #     pro = tf.nn.softmax(out.logits)
    #     label_index = np.argmax(pro, axis=-1)
    #     label = label_encode.inverse_transform(label_index)
    #     score = np.take_along_axis(pro.numpy(), np.expand_dims(label_index, axis=-1), axis=1)
    #     for i, l, p in zip(guids, label, score):
    #         feature_ids_map[i] = [l, p.tolist()]
    #
    # with open(args.output_file, "w", encoding="utf-8") as g:
    #     for k, v in feature_ids_map.items():
    #         g.write(k + "\t" + json.dumps(v, ensure_ascii=False) + "\n")

    # 使用tensorflow datasets进行分布式预测
    data = tf.data.Dataset.from_generator(
        lambda: [(feature.input_ids, feature.segment_ids, [feature.guid]) for feature in features],
        output_types=(tf.int32, tf.int32, tf.string)

    )
    # data = data.batch(batch_size=2)
    distribute_ds = data.padded_batch(2, padded_shapes=([None], [None], [None]), drop_remainder=False)


    def replica_fn(inputs):
        out = model(inputs)
        pro = tf.nn.softmax(out.logits)
        label_index = tf.argmax(pro, axis=-1)
        return label_index, pro


    feature_ids_map = {}

    for (input_ids, token_type_ids, guids) in distribute_ds:
        distribute_out = strategy.run(replica_fn, args=((input_ids, token_type_ids),))
        label_concat = tf.concat(distribute_out[0].values, axis=0)
        score_concat = tf.concat(distribute_out[1].values, axis=0)
        guids_concat = tf.concat(guids.values, axis=0)
        # ###########合并多GPU预测结果
        label = label_encode.inverse_transform(label_concat.numpy())
        score = np.take_along_axis(score_concat.numpy(), np.expand_dims(label_concat, axis=-1), axis=1)
        for i, l, p in zip(guids_concat.numpy(), label, score):
            feature_ids_map[i] = [l, p.tolist()]

