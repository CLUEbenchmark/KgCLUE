#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/27 14:41
# @Author  : 刘鑫
# @FileName: demo.py
# @Software: PyCharm
from elasticsearch import Elasticsearch
from entity_extract import EntityExtract
from SIM.sim_predict import SimPredict

NER_MODEL_PATH = "../NER/model"
ee = EntityExtract(NER_MODEL_PATH)
SIM_MODEL_PATH = "../SIM/model"
sim = SimPredict(SIM_MODEL_PATH)

es_host = "127.0.0.1"
es_port = "9200"
es = Elasticsearch([":".join((es_host, es_port))])

sentence = "巫山县疾病预防控制中心的机构职能是什么?"

entity = "".join(ee.extract(sentence))

body = {
    "query": {
        "term": {
            "entity.keyword": entity
        }
    }
}
es_results = es.search(index="kbqa-data", doc_type="kbList", body=body, size=30)
attribute_list, answer_list = list(), list()
for i in range(len(es_results['hits']['hits'])):
    relation = es_results['hits']['hits'][i]['_source']['relation']
    value = es_results['hits']['hits'][i]['_source']['value']
    attribute_list.append(relation)
    answer_list.append(value)

for attribute, answer in zip(attribute_list, answer_list):
    isAttribute = sim.predict_one(sentence, attribute, EVAL_MODE=True)
    if isAttribute:
        print("问题：%s，属性：%s,回答：%s" % (sentence, attribute, answer))
