#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/27 14:56
# @Author  : 刘鑫
# @FileName: entity_extract.py
# @Software: PyCharm
import json

from NER.ner_predict import NerPredict
from frame.bert import tokenization
from config import config


class EntityExtract(object):
    def __init__(self, MODEL_PATH):
        self.ner = NerPredict(MODEL_PATH)
        self.tokenizer_ = tokenization.FullTokenizer(vocab_file=config.vocab_file)
        pass

    def extract(self, sentence):
        def _merge_WordPiece_and_single_word(entity_sort_list):
            entity_sort_tuple_list = []
            for a_entity_list in entity_sort_list:
                entity_content = ""
                entity_type = None
                for idx, entity_part in enumerate(a_entity_list):
                    if idx == 0:
                        entity_type = entity_part
                        if entity_type[:2] not in ["B-", "I-"]:
                            break
                    else:
                        if entity_part.startswith("##"):
                            entity_content += entity_part.replace("##", "")
                        else:
                            entity_content += entity_part
                if entity_content != "":
                    entity_sort_tuple_list.append((entity_type[2:], entity_content))
            return entity_sort_tuple_list

        ner_out = self.ner.predict_one(sentence, EVAL_MODE=True)[1:]

        seq_token = []
        lenth = len(sentence)

        for idx, n in enumerate(ner_out):
            if idx <= lenth and n != "[SEP]":
                seq_token.append(n)

        entity_sort_list = []
        entity_part_list = []
        token_in_not_UNK = self.tokenizer_.tokenize(sentence)
        for idx, token_label in enumerate(seq_token):
            if token_label == "O":
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
            if token_label.startswith("B-"):
                if len(entity_part_list) > 0:  # 适用于 B- B- *****的情况
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
                entity_part_list.append(token_label)
                entity_part_list.append(token_in_not_UNK[idx])
                if idx == len(seq_token) - 1:
                    entity_sort_list.append(entity_part_list)
            if token_label.startswith("I-") or token_label == "[##WordPiece]":
                if len(entity_part_list) > 0:
                    entity_part_list.append(token_in_not_UNK[idx])
                    if idx == len(seq_token) - 1:
                        entity_sort_list.append(entity_part_list)

        entity_sort_tuple_list = _merge_WordPiece_and_single_word(entity_sort_list)

        entitys = []
        for entity in entity_sort_tuple_list:
            if entity[0] == "NP":
                entitys.append(entity[1])
        return entitys


if __name__ == '__main__':
    MODEL_PATH = "../NER/model"
    ee = EntityExtract(MODEL_PATH)
    ff = open("./out.txt", 'w', encoding='utf-8')
    with open(r"C:\Users\11943\Documents\GitHub\RoBERTa-wwm-ext\raw_data\dev.json", "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line:
                line = json.loads(line)
                text = line["question"]
                answer = line["answer"]
                entity = answer.split("|||")[0].split("（")[0]
                p_entity = ee.extract(text)
                ff.write(entity + "\t" + " ".join(p_entity) + "\n")
            else:
                break
    ff.close()
