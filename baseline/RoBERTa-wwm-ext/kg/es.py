# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :    deploy es
   Author :         nijinxin
   Date :           2021-11-24
-------------------------------------------------

'''

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def deployES(input_file, host, port):
    """ put triples into es
        input: input_file, host, port
        output: None
    """
    es_dir = ":".join((host, port))
    print("es_dir:",es_dir)
    es = Elasticsearch([es_dir])
    es.indices.create(index='kbqa-data',ignore=[400, 404])
    f = open(input_file, 'r', encoding='utf-8')
    
    id_num = 0
    lines = []
    while True:
        line = f.readline()
        id_num += 1
        if line == "":
            break
        line = line.split("\t")
        if len(line[0].split("（")) == 2 and line[0][-1] == "）":
            entity = line[0].split("（")[0]
            ambiguity = line[0].split("（")[1].strip("）")

        else:
            entity = line[0]
            ambiguity = "None"
        
        action = {
                "_index": "kbqa-data",
                "_type": "kbList",
                "_id": id_num, #_id 也可以默认生成，不赋值
                "_source": {
                    "entity": entity,
                    "relation": line[1],
                    "value": line[2].strip(),
                    "ambiguity": ambiguity}}
        lines.append(action)
        if id_num % 5000 == 0:
            
            bulk(es, lines, index="kbqa-data", raise_on_error=True)
            lines = []
        if id_num % 50000 == 0:
            print(id_num, " triples has been deployed")
    f.close()
    

if __name__ == "__main__":
    input_file = "../knowledge/Knowledge.txt"
    host = "127.0.0.1"
    port = "9200"
    deployES(input_file, host, port) 