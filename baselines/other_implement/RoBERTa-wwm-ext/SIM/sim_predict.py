#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 10:42
# @Author  : 刘鑫
# @FileName: seq_predict.py
# @Software: PyCharm

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from frame.bert import tokenization
from config import config
from sim_data_making import SimDataMaking
from utils import id2label


# 实体识别预测类
class NerPredict(object):

    def __init__(self, MODEL_PATH):
        self.model_path = MODEL_PATH

        self.seq_id2label = id2label(config.token_label_list)

        # 准备token
        self.tokenizer_ = tokenization.FullTokenizer(vocab_file=config.vocab_file)
        attribute_path = "../raw_data/train.json"
        fake_example_nums = 20
        self.data_making = SimDataMaking(attribute_path, fake_example_nums, do_lower_case=True,
                                         max_seq_length=config.sim_max_seq_length)
        self.sess = self.load_model()

        self.input_ids = self.sess.graph.get_tensor_by_name("input_ids:0")
        self.input_mask = self.sess.graph.get_tensor_by_name("input_mask:0")
        self.segment_ids = self.sess.graph.get_tensor_by_name("segment_ids:0")
        self.keep_prob = self.sess.graph.get_tensor_by_name("keep_prob:0")
        # 预测的结果
        self.p = self.sess.graph.get_tensor_by_name("loss/Cast_1:0")

    def load_model(self):
        try:
            checkpoint = tf.train.get_checkpoint_state(self.model_path)
            input_checkpoint = checkpoint.model_checkpoint_path
            print("[INFO] input_checkpoint:", input_checkpoint)
        except Exception as e:
            input_checkpoint = self.model_path
            print("[INFO] Model folder", self.model_path, repr(e))

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True
        tf.reset_default_graph()
        # We import the meta graph and retrieve a Saver
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We start a session and restore the graph weights
        sess_ = tf.Session()
        saver.restore(sess_, input_checkpoint)

        return sess_

    def predict_one(self, PREDICT_TXT, attribute, EVAL_MODE=False):
        feature = self.data_making.convert_single_example(PREDICT_TXT, attribute, test_mode=True)

        # def convert_single_example(self, text, labeling_list=[], test_mode=False):
        def id2label_f(id_list):
            predictions = []
            if EVAL_MODE:
                for id in id_list:
                    predictions.append(self.seq_id2label[id])
            else:
                for id in id_list:
                    predictions.append(self.seq_id2label[id])
                    if id == 4:
                        break
            return predictions

        feed = {self.input_ids: [feature[0]],
                self.input_mask: [feature[1]],
                self.segment_ids: [feature[2]],
                self.keep_prob: 1.0}

        probs = self.sess.run(self.p, feed)
        for tmp in probs:
            # out = id2label_f(tmp)
            tmp = list(tmp)
            if tmp[1] == 1:
                out = True
            else:
                out = False
        return out


if __name__ == '__main__':
    MODEL_PATH = "./model"

    PREDICT_TXT = "东瓯王发生的主要事件是什么？"
    # attribute = "你好"

    attributes = ["东瓯王后", "中文名", "主要事件", "代表人物", "姓氏", "所属地", "中文名", "位置"]
    label = [False, False, True, False, False, False, False, False]
    ner = NerPredict(MODEL_PATH)
    outs = []
    for attribute in attributes:
        out = ner.predict_one(PREDICT_TXT, attribute, EVAL_MODE=True)
        outs.append(out)
    print(outs)