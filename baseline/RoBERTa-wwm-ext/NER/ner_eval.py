#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/20 10:54
# @Author  : 刘鑫
# @FileName: seq_eval.py
# @Software: PyCharm
import json
import os

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report

from ner_predict import NerPredict
from ner_data_making import NerDataMaking
from config import config


# 评估说明：既然是关注序列标注模型的分类效果，文本分类结果应该给予正确的

class NerEval(object):
    def __init__(self, Ner_MODEL_PATH):
        self.ner_data_make = NerDataMaking(do_lower_case=True, max_seq_length=config.max_seq_length)
        self.ner = NerPredict(Ner_MODEL_PATH)
        self.seq_id2label = self.ner.seq_id2label

    def id2label_f(self, id_list):
        predictions = []
        for id in id_list:
            predictions.append(self.seq_id2label[id])
        return predictions

    def do_eval(self, data_files=["../raw_data/test.json"]):

        for data_file in data_files:
            y_true = []
            y_pred = []
            with open(data_file, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line:
                        line = json.loads(line)
                        text = line["question"]
                        answer = line["answer"]
                        entity = answer.split("|||")[0].split("（")[0]

                        token_label = self.ner_data_make.text2label(text, entity)
                        feature = self.ner_data_make.convert_single_example(text, token_label)
                        label = self.id2label_f(feature.label_ids)
                        predict_label = self.ner.predict_one(text, EVAL_MODE=True)

                        y_true.append(label)
                        y_pred.append(predict_label)

                    else:
                        break

            acc_s = accuracy_score(y_true, y_pred)
            precision_s = precision_score(y_true, y_pred)
            recall_s = recall_score(y_true, y_pred)
            f1_s = f1_score(y_true, y_pred)
            report = classification_report(y_true, y_pred)

            print(f'\t\t准确率为： {acc_s}')
            print(f'\t\t查准率为： {precision_s}')
            print(f'\t\t召回率为： {recall_s}')
            print(f'\t\tf1值为： {f1_s}')
            print(report)

            file_name = data_file.split("/")[-1].split(".")[0]
            if not os.path.exists("./out/NER/"):
                os.makedirs("./out/NER/")
            with open(os.path.join("./out/NER/", file_name + "_out.txt"), "w", encoding='utf-8') as f:
                print("score will be stored in %s_out.txt" % file_name)
                f.write(f'准确率为： {acc_s}')
                f.write(f'查准率为： {precision_s}')
                f.write(f'召回率为： {recall_s}')
                f.write(f'f1值为： {f1_s}')
                f.write(f'{report}')


if __name__ == '__main__':
    Ner_MODEL_PATH = "./model"
    ner_eval = NerEval(Ner_MODEL_PATH)
    ner_eval.do_eval(data_files=["../raw_data/test_public.json", "../raw_data/eval.json", "../raw_data/train.json"])
# "../raw_data/train.json", "../raw_data/eval.json",
