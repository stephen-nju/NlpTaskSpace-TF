# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: dataset.py
@time: 2022/3/11 11:40
"""
import csv
from functools import partial
from multiprocessing import cpu_count, Pool

import tensorflow as tf
from tqdm import tqdm


class InputExampleBase(object):
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


class InputFeaturesBase(object):
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


class DataProcessorBase(object):
    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_path):
        """See base class."""
        raise NotImplementedError

    def get_dev_examples(self, data_path):
        """See base class."""
        raise NotImplementedError

    def get_test_examples(self, data_path):
        raise NotImplementedError

    @staticmethod
    def get_labels(path=None):
        raise NotImplementedError

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file,"r",encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        # with open(input_file, "r", encoding="utf-8") as f:
        #     reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        #     lines = []
        #     for line in reader:
        #         lines.append(line)
        #     return lines

    @staticmethod
    def convert_single_example(example, tokenizer, label_encode, max_length):
        raise NotImplementedError

    def convert_examples_to_features(self, examples, tokenizer, label_encode, max_length, threads=4):
        threads = min(threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(
                self.convert_single_example,
                tokenizer=tokenizer,
                label_encode=label_encode,
                max_length=max_length
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert examples to features",
                )
            )
        return features
