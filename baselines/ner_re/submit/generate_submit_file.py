import requests, json
import sys
import re
sys.path.append("../../../baselines/")
try:
    from baselines.ner_re.sim.predict import Predictor
except Exception as e:
    from sim.predict import Predictor
import argparse

"""
生成预测文件
1.获取实体预测文件
2.得到的实体
3.根据实体，利用es找到候选的关系列表
4.利用相似性模型。使用问句和候选关系列表，得到最相关的关系
5.利用实体、关系，定位到最终的答案
"""

def contain_english(strings):
    # 判断是否是英文
    english_re = re.compile(r'[A-Za-z]', re.S)
    english_list = re.findall(english_re, strings)
    if len(english_list)>0:
        return True
    else:
        return False

def generate_submit_file_fn(test_file,ner_predict_file,target_file,log_file):
    # 1.获取实体预测文件
    test_file_object=open(test_file,'r',encoding='utf-8')
    test_lines=test_file_object.readlines()
    target_object=open(target_file,'w',encoding='utf-8')
    log_object=open(log_file,'w',encoding='utf-8')

    predict_ner_test_object=open(ner_predict_file,'r')
    predict_ner_lines=predict_ner_test_object.readlines()
    print("test_lines:",len(test_lines),";predict_ner_lines:",len(predict_ner_lines))
    question_entity_list=[]
    # 2. 得到的实体
    entity_empty=0
    for i,line in enumerate(test_lines):
        # line= {"id": 0, "question": "我很好奇刘质平的老师是谁？"}
        # line={"id": 0, "tag_seq": "O O O O B-NER I-NER I-NER O O O O O O", "entities": [["NER", 4, 6]]}
        question=json.loads(line.strip())["question"]
        json_data=json.loads(predict_ner_lines[i])
        entities_with_indices=json_data["entities"] # ['NER', 4, 6]
        if len(entities_with_indices)==0:
            question_entity_list.append([question, ''])
            entity_empty=entity_empty+1
            continue
        #print(i,"entities_with_indices:",entities_with_indices)
        entities_with_indices=entities_with_indices[0]
        start_index, end_index=entities_with_indices[1],entities_with_indices[2]
        entity=question[start_index:end_index+1]

        #################################################
        if contain_english(entity) and " " in entity and contain_english(question[end_index+1:end_index+2]): # 实体包含英文，并且包含字母并且下一个字符也是英文 ==> 修正一个模型错误即预测时忽略了空格的位置（可能是由于数据处理原因引起）
            entity=question[start_index:end_index+2]
        #################################################
        question_entity_list.append([question, entity])
        if i <10:
            print(i, "json_data:", json_data)
            print("entities_with_indices:", entities_with_indices)
            print("entity:",entity,";question:",question,";start_index:",start_index,";end_index:",end_index,";predict_ner_line:")

    print("entity_empty:",entity_empty)
    # 3.根据实体，利用es找到候选的关系列表
    for ii, element in enumerate(question_entity_list):
        question, entity = element
        if entity!='':
            attribute_list, answer_list, ambiguity_dict, original_entity_list=get_candidate_list_by_api(entity)
            # 4.利用相似性模型。使用问句和候选关系列表，得到最相关的关系
            if len(attribute_list)>0:
                index,attribute=get_attribute(question,attribute_list)
                # 5.利用实体、关系，定位到最终的答案
                answer=answer_list[index]
                entity_additional=original_entity_list[index]
                # if ii<10:
                print(ii,"question:",question,";entity:",entity,";answer:",answer,";entity_additional:",entity_additional)
                print(ii,"ambiguity_dict:",ambiguity_dict)

                if entity_additional.isdigit()==False: # 不是数字即是文本，那么说明entity_additional中有内容，则使用entity_additional的内容。
                    entity=entity+"（"+entity_additional+"）"

                json_string={"id":ii,"answer":entity.strip()+" ||| "+attribute.strip()+" ||| "+answer.strip()}
            else:
                json_string = {"id": ii, "answer": ""}
                # 写日志
                # print(ii,type(attribute_list),attribute_list)
                log_json={"id":ii,"question":question,"entity:":entity,"attribute_list:":str(attribute_list)}
                log_json = json.dumps(log_json, ensure_ascii=False)
                log_object.write(log_json+"\n")
        else:
            json_string={"id":ii,"answer":""}
            # 写日志
            log_json = {"id":ii,"question:":question}
            log_json = json.dumps(log_json, ensure_ascii=False)
            log_object.write(log_json + "\n")
        json_string = json.dumps(json_string, ensure_ascii=False)
        target_object.write(json_string + "\n")
    print("submit file generated. path:"+str(target_file))
    target_object.close()
    log_object.close()

parser = argparse.ArgumentParser(description='manual to this script')
base_model_path = 'prev_trained_model/chinese_rbt3_L-3_H-768_A-12/'
# parser.add_argument('--gpu_id', type=str, default = "7") # -1
parser.add_argument('--num_classes', type=str, default = 2)
parser.add_argument('--maxlen', type=str, default = 64)
parser.add_argument('--output_dir', type=str, default="outputs/kg_sim_output")
parser.add_argument('--config_path', type=str, default=base_model_path+"bert_config_rbt3.json")
parser.add_argument('--checkpoint_path', type=str, default=base_model_path+"bert_model.ckpt")
parser.add_argument('--dict_path', type=str, default=base_model_path+"vocab.txt")
args = parser.parse_args()
predictor = Predictor(args)
def get_attribute(question,attribute_list):
    """
    利用相似性模型。使用问句和候选关系列表，得到最相关的关系
    :param question:
    :param attribute_list:
    :return:
    """
    p_best=0.0
    index_best=-1
    # print("question:",question,";attribute_list:",attribute_list)
    for index,attr in enumerate(attribute_list):
        possibility = predictor.predict(question, attr)
        # print("question:",question,"attr",attr,";possibility:",possibility)
        if possibility>p_best:
            p_best=possibility
            index_best=index
    # print("-------------------------------")
    attribute=attribute_list[index_best]
    return index_best,attribute

def get_candidate_list_by_api(entity):
    """
    调用接口，根据实体得到候选关系、候选答案、候选的实体的释义
    :param entity:
    :return:
    """
    data = {'entity': entity}
    url = 'http://47.75.32.69:5004/search_kb/entity/'
    r = requests.post(url, data=json.dumps(data))
    # print("r:",r.text)
    data = json.loads(r.text)["data"]
    attribute_list=data["attribute_list"]
    answer_list=data["answer_list"]
    ambiguity_dict=data["ambiguity_dict"]
    original_entity_list=data["original_entity_list"]
    return attribute_list,answer_list,ambiguity_dict,original_entity_list

test_file='../../datasets/test.json' # KgCLUE测试集
ner_predict_file='outputs/kg_output/bert/test_prediction.json' # NER命名实体识别的预测文件
target_file='kgclue_predict_rbt3.json'
log_file='./error_log.txt'
generate_submit_file_fn(test_file,ner_predict_file,target_file,log_file)