#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 15:43
# @Author  : 刘鑫
# @FileName: utils.py
# @Software: PyCharm

import random


def _index_q_list_in_k_list(q_list, k_list):
    """Known q_list in k_list, find index(first time) of q_list in k_list"""
    q_list_length = len(q_list)
    k_list_length = len(k_list)
    for idx in range(k_list_length - q_list_length + 1):
        t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
        if all(t):
            idx_start = idx
            return idx_start


class NerInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


def id2label(label_list):
    out = dict()
    for idx, label in enumerate(label_list):
        out[idx] = label
    return out


def getAttribute(Attributes, attribute):
    tmp = Attributes[-1]
    if attribute in Attributes:
        Attributes.remove(attribute)
    else:
        Attributes.remove(tmp)

    x = random.randint(0, abs(len(Attributes) - 1))
    if attribute in Attributes:
        Attributes.append(attribute)
    else:
        Attributes.append(tmp)
    return Attributes[x]


from config import config

# print(getAttribute(config.Attributes, "出品公司"))
