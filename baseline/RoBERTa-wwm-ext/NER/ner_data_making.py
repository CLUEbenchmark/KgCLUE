#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 14:33
# @Author  : 刘鑫
# @FileName: ner_data_making.py
# @Software: PyCharm
import collections
import json
import os
import tensorflow as tf

from frame.bert import tokenization
from config import config
from utils import _index_q_list_in_k_list, NerInputFeatures


class NerDataMaking(object):
    def __init__(self, do_lower_case=True, max_seq_length=128):
        self.task_name = "NER"
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file,
                                                         do_lower_case=do_lower_case)  # 初始化 bert_token 工具

        self.max_seq_length = max_seq_length

    def makdir(self, OUTPUT_DIR, DIR_NAME):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), OUTPUT_DIR)):
            os.makedirs(os.path.join(os.path.dirname(__file__), OUTPUT_DIR))
        if not os.path.exists(os.path.join(os.path.dirname(__file__), OUTPUT_DIR, DIR_NAME)):
            os.makedirs(os.path.join(os.path.dirname(__file__), OUTPUT_DIR, DIR_NAME))

    def text2label(self, text, entity):
        text_token = self.bert_tokenizer.tokenize(text)
        entity_token = self.bert_tokenizer.tokenize(entity)
        entity_lenth = len(entity_token)
        idx_start = _index_q_list_in_k_list(entity_token, text_token)
        labeling_list = ["O"] * len(text_token)  # 先用全O覆盖
        if idx_start is None:
            tokener_error_flag = True
            # self.bert_tokener_error_log_f.write(subject_object + " @@ " + text + "\n")
        else:
            labeling_list[idx_start] = "B-NP"
            if entity_lenth == 2:
                labeling_list[idx_start + 1] = "I-NP"
            elif entity_lenth >= 3:
                labeling_list[idx_start + 1: idx_start + entity_lenth] = ["I-NP"] * (entity_lenth - 1)

        for idx, token in enumerate(text_token):
            if token.startswith("##"):
                labeling_list[idx] = "[##WordPiece]"

        return labeling_list

    def convert_single_example(self, text, labeling_list=[], test_mode=False):

        token_label_list = config.token_label_list

        token_label_map = {}
        for (i, label) in enumerate(token_label_list):
            token_label_map[label] = i

        text_token = self.bert_tokenizer.tokenize(text)

        # Account for [CLS] and [SEP] with "- 2"
        if len(text_token) > self.max_seq_length - 2:
            text_token = text_token[0:(self.max_seq_length - 2)]

        tokens = []
        token_label_ids = []
        segment_ids = []
        # 添加起始位置
        tokens.append("[CLS]")
        segment_ids.append(0)
        token_label_ids.append(token_label_map["[CLS]"])

        if test_mode:
            labeling_list = ["O"] * len(text_token)

        for token, label in zip(text_token, labeling_list):
            tokens.append(token)
            segment_ids.append(0)
            token_label_ids.append(token_label_map[label])

        tokens.append("[SEP]")
        segment_ids.append(0)
        token_label_ids.append(token_label_map["[SEP]"])  # 第一句话结束

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            token_label_ids.append(0)
            tokens.append("[Padding]")

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(token_label_ids) == self.max_seq_length

        if test_mode:
            feature = (input_ids, input_mask, segment_ids)
        else:
            feature = NerInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=token_label_ids,
                is_real_example=True)
        # print(input_ids)
        # print(input_mask)
        # print(segment_ids)
        # print(token_label_ids)
        return feature

    def input2output(self, INPUT_DATA_PATHS, OUTPUT_DIR):
        for data_path in INPUT_DATA_PATHS:
            file_name = data_path.split('/')[-1]
            file_type = file_name.split('.')[0]
            self.makdir(OUTPUT_DIR, file_type)  # 生成文件名对应的文件夹
            OUT_PATH = os.path.join(os.path.dirname(__file__), OUTPUT_DIR, file_type)
            # 创建需要写入的文件
            text_f = open(os.path.join(OUT_PATH, "text.txt"), "w", encoding='utf-8')
            token_in_f = open(os.path.join(OUT_PATH, "token_in.txt"), "w", encoding='utf-8')
            # token_in_not_UNK_f = open(os.path.join(OUT_PATH, "token_in_not_UNK.txt"), "w", encoding='utf-8')
            token_label_f = open(os.path.join(OUT_PATH, "token_label.txt"), "w", encoding='utf-8')
            tf_writer = tf.python_io.TFRecordWriter(os.path.join(OUT_PATH, file_type + ".tf_record"))
            with open(data_path, "r", encoding='utf-8') as f:
                count_numbers = 0
                while True:
                    line = f.readline()
                    if line:
                        line = json.loads(line)
                        text = line["question"]
                        answer = line["answer"]
                        entity = answer.split("|||")[0].split("（")[0]
                        token_label = self.text2label(text, entity)

                        # 写入txt文件
                        text_f.write(text + "\n")
                        token_in_f.write(" ".join(self.bert_tokenizer.tokenize(text)) + "\n")
                        # token_in_not_UNK_f.write(" ".join(self.bert_tokenizer.tokenize_not_UNK(text)) + "\n")
                        token_label_f.write(" ".join(token_label) + "\n")

                        # 写入tf_record
                        feature = self.convert_single_example(text, token_label)

                        def create_int_feature(values):
                            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                            return f

                        features = collections.OrderedDict()
                        features["input_ids"] = create_int_feature(feature.input_ids)
                        features["input_mask"] = create_int_feature(feature.input_mask)
                        features["segment_ids"] = create_int_feature(feature.segment_ids)
                        features["label_ids"] = create_int_feature(feature.label_ids)
                        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

                        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                        tf_writer.write(tf_example.SerializeToString())
                        count_numbers += 1
                        if count_numbers % 10000 == 0:
                            print("Writing example %d " % (count_numbers))
                        # break
                    else:
                        break
            print("all numbers", count_numbers)
            text_f.close()
            token_in_f.close()
            # token_in_not_UNK_f.close()
            token_label_f.close()
            tf_writer.close()


if __name__ == '__main__':
    INPUT_DATA_PATHS = ["raw_data/train.json","raw_data/eval.json","raw_data/test_public.json"]
    OUTPUT_DIR = "data"
    ner_data_make = NerDataMaking(do_lower_case=True, max_seq_length=config.max_seq_length)
    ner_data_make.input2output(INPUT_DATA_PATHS, OUTPUT_DIR)
