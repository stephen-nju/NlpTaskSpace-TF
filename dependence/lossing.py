# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: lossing.py
@time: 2021/9/29 16:24
"""
import tensorflow as tf
from tensorflow_addons.utils.keras_utils import is_tensor_or_variable


class LossFunctionWrapper(tf.keras.losses.Loss):
    def __init__(
            self, fn, reduction=tf.keras.losses.Reduction.AUTO, name=None, **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in iter(self._fn_kwargs.items()):
            config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
        base_config = super().get_config()
        return {**base_config, **config}


class AsymmetricLoss(LossFunctionWrapper):

    def __init__(self,
                 from_logits=False,
                 gamma_pos=1,
                 gamma_neg=4,
                 clip=0.05,
                 reduction=tf.keras.losses.Reduction,
                 name='asymmetric_loss'):
        super().__init__(
            multi_label_asymmetric_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
            eps=1e-8
        )
        self.from_logits = from_logits


"""

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
"""


def multi_label_asymmetric_loss(y_true, y_pred, from_logits, gamma_pos=1, gamma_neg=4, clip=0.05, eps=1e-8):
    # Calculating Probabilities
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    if from_logits:
        y_pred_prob = tf.nn.sigmoid(y_pred)
    else:
        y_pred_prob = y_pred

    logits_sigmoid_pos = y_pred_prob
    logits_sigmoid_neg = 1 - logits_sigmoid_pos

    # asymmetric clipping
    if clip is not None and clip > 0:
        # logits_sigmoid_neg + clip 有可能大于1
        # 负样本概率需要做平移
        logits_sigmoid_neg = tf.clip_by_value((logits_sigmoid_neg + clip), clip_value_min=0, clip_value_max=1.0)

    # basic cross entropy
    # logits_sigmoid_pos的取值范围是0-1，因此可以直接可以取对数log，不会溢出
    loss_pos = y_true * tf.math.log(tf.clip_by_value(logits_sigmoid_pos, clip_value_min=eps, clip_value_max=1.0))
    loss_neg = (1 - y_true) * tf.math.log(tf.clip_by_value(logits_sigmoid_neg, clip_value_min=eps, clip_value_max=1.0))

    loss = loss_pos + loss_neg
    # Asymmetric focusing
    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = logits_sigmoid_pos * y_true
        pt1 = logits_sigmoid_neg * (1 - y_true)
        pt = pt0 + pt1

        # pt = y_true * y_pred_prob + (1 - y_true) * (1 - y_pred_prob)

        one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
        one_sided_w = tf.pow(1 - pt, one_sided_gamma)

        one_sided_w_no_gradient = tf.stop_gradient(tf.pow(1 - pt, one_sided_gamma))
        # one_sided_w_no_gradient = tf.stop_gradient([pt0, pt1, pt, one_sided_gamma, one_sided_w])

        # one_sided_w_no_gradient = tf.stop_gradient(one_sided_w])
        loss *= one_sided_w_no_gradient

    return -tf.reduce_sum(loss)


def single_label_asymmetric_loss(labels, logits, gamma_pos=4.0, gamma_neg=0.0, eps: float = 0.1, reduction="mean"):
    """
    :param labels: tf.tensor, sparse_label_encoder， 不是one_hot编码
    :param logits:
    :param gamma_pos:
    :param gamma_neg:
    :param eps:
    :param reduction:
    :return:
    """

    num_classes = logits.get_shape().as_list()[-1]

    log_probs = tf.nn.log_softmax(logits)
    shape = labels.get_shape().as_list()
    origin_target_classes = tf.one_hot(labels, depth=num_classes)

    log_probs = tf.nn.log_softmax(logits)

    # asymmetric loss weights
    target_classes = origin_target_classes
    anti_targets = 1 - target_classes

    logits_pos = tf.exp(log_probs)

    logits_neg = 1 - logits_pos
    print("logits_pos: ", logits_pos)
    print("target_classes: ", target_classes)

    logits_pos = tf.multiply(logits_pos, target_classes)
    logits_neg = tf.multiply(logits_neg, anti_targets)

    print("logits_pos: ", logits_pos)

    # logits_pos *= target_classes
    # logits_neg *= anti_targets

    asymmetric_w = tf.pow(1 - logits_pos - logits_neg, gamma_pos * target_classes + gamma_neg * anti_targets)

    log_probs = log_probs * asymmetric_w

    if eps > 0:  # label smoothing
        origin_target_classes = origin_target_classes * (1 - eps) + eps / num_classes

    # loss calculation

    loss = -tf.reduce_sum(tf.multiply(origin_target_classes, log_probs), axis=-1)

    if reduction == "mean":
        loss = tf.reduce_mean(loss)

    return loss


class SymmetricLoss(LossFunctionWrapper):
    def __init__(self,
                 from_logits=False,
                 alpha=1,
                 beta=1,
                 reduction=tf.keras.losses.Reduction,
                 name='symmetric_loss'

                 ):
        super().__init__(
            symmetric_cross_entropy_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            alpha=alpha,
            beta=beta
        )
        self.from_logits = from_logits


def symmetric_cross_entropy_loss(y_true, y_pred, from_logits, alpha, beta, clip):
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)

    y_true_1 = y_true
    y_pred_1 = y_pred

    y_true_2 = y_true
    y_pred_2 = y_pred
    if clip is not None and clip > 0:
        y_pred_1 = tf.clip_by_value(y_pred_1, clip, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, clip, 1.0)

    return alpha * tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.math.log(y_pred_1), axis=-1)) + beta * tf.reduce_mean(
        -tf.reduce_sum(y_pred_2 * tf.math.log(y_true_2), axis=-1))
