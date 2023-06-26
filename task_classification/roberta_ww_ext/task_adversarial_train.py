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
from tensorflow.python.keras.engine import data_adapter
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


# def creat_FGM(epsilon=1.0):
#     @tf.function
#     def train_step(self, data):
#     '''
#     计算在embedding上的gradient
#     计算扰动 在embedding上加上扰动
#     重新计算loss和gradient
#     删除embedding上的扰动，并更新参数
#     '''
#         data = data_adapter.expand_1d(data)
#         x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
# 	    with tf.GradientTape() as tape:
# 	        y_pred = model(x,training=True)
# 	        loss = loss_func(y,y_pred)
# 	    embedding = model.trainable_variables[0]
# 	    embedding_gradients = tape.gradient(loss,[model.trainable_variables[0]])[0]
# 	    embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
# 	    delta = 0.2 * embedding_gradients / (tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + 1e-8)  # 计算扰动
# 	    model.trainable_variables[0].assign_add(delta)
# 	    with tf.GradientTape() as tape2:
# 	        y_pred = model(x,training=True)
# 	        new_loss = loss_func(y,y_pred)
# 	    gradients = tape2.gradient(new_loss,model.trainable_variables)
# 	    model.trainable_variables[0].assign_sub(delta)
# 	    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
# 	    train_loss.update_state(loss)
# 	    return {m.name: m.result() for m in self.metrics}
#     return train_step

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


# self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
# self.compiled_metrics.update_state(y, y_pred, sample_weight)

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
        # os.removedirs(log_dir)
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
        model = TFBertAdversarialTrain.from_pretrained(args.bert_model,
                                                       config=config
                                                       )
        lr_schedule = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=args.lr,
            maximal_learning_rate=3e-5,
            step_size=800,
            scale_fn=lambda x: 1.,
            scale_mode="cycle",
            name="MyCyclicScheduler")

        # 使用Hook方式，替换父类中的方法

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
