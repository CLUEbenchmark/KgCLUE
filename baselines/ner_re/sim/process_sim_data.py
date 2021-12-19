# coding=UTF-8
import json
import random
import os
"""
生成相似度模型(SIM)需要的训练数据
"""
num_negative_examples=8
def generate_train_data(source_file, target_file, target_file2=None):
    source_lines=open(source_file,'r').readlines()
    target_object=open(target_file,'w')
    target_object2=''
    if target_file2 is not None:
        target_object2=open(target_file2,'w')
    example_list=[]
    # 1.正样本(label=1)
    relationship_list=[]
    for i, line in enumerate(source_lines):
        # line= {"id": 15, "question": "哈药的员工有多少人啊？", "answer": "哈药 ||| 职工 ||| 2.01万人"}
        json_raw=json.loads(line.strip())
        question=json_raw["question"]
        relationship=json_raw["answer"].split(" ||| ")[1].strip() #
        label="1"
        json_result={"question": question,"relationship":relationship,"label":label}
        example_list.append(json_result)
        relationship_list.append(relationship)
        if i<5:
            print("json_result:",json_result)


    # 2.负样本(label=0)
    relationship_list=list(set(relationship_list))
    for i, line2 in enumerate(source_lines):
        json_raw=json.loads(line2.strip())
        question2=json_raw["question"]
        for ddd in range(num_negative_examples):
            relationship_random=random.choice(relationship_list)
            label = "0"
            json_result = {"question": question2, "relationship": relationship_random, "label": label}
            example_list.append(json_result)

    # 3.写入文件
    random.shuffle(example_list)
    num_examples=len(example_list)
    for index, element in enumerate(example_list):
        json_result=json.dumps(element,ensure_ascii=False)

        if target_file2 is not None: # 不让target_file2不为空，并且属于前一半的数据，那么分配给target_file2（即test）;默认分配给dev
            if index<int(num_examples*0.50):
                target_object2.write(json_result+"\n")
                continue
        target_object.write(json_result+"\n")

    target_object.close()
    print("process similiarity data finished. file_path:",target_file)

# 以下代码生成相似度模型需要的训练集
DATA_DIR='../../datasets/'
PROCESS_DIR='./processed_data/'
source_file=DATA_DIR+'train.json'
target_file=PROCESS_DIR+'train.json'

if not os.path.exists(PROCESS_DIR):
    os.mkdir(PROCESS_DIR)

# 生成NER的训练集
generate_train_data(source_file, target_file)

# 生成NER的验证集和测试集
source_file=DATA_DIR+'dev.json'
target_file=PROCESS_DIR+'dev.json'
target_file2=PROCESS_DIR+'test.json'
generate_train_data(source_file, target_file,target_file2=target_file2)

