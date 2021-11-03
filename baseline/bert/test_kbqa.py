from BERT_CRF import BertCrf
from NER import NerProcessor, CRF_LABELS
from SIM import SimProcessor,SimInputFeatures
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import pymysql
from tqdm import tqdm, trange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_ner_model(config_file,pre_train_model,label_num = 2):
    model = BertCrf(config_name=config_file,num_tags=label_num, batch_first=True)
    model.load_state_dict(torch.load(pre_train_model))
    return model.to(device)


def get_sim_model(config_file,pre_train_model,label_num = 2):
    bert_config = BertConfig.from_pretrained(config_file)
    bert_config.num_labels = label_num
    model = BertForSequenceClassification(bert_config)
    model.load_state_dict(torch.load(pre_train_model))
    return model


def get_entity(model,tokenizer,sentence,max_len = 64):
    pad_token = 0
    sentence_list = list(sentence.strip().replace(' ',''))
    text = " ".join(sentence_list)
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        truncate_first_sequence=True  # We're truncating the first sequence in priority if True
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    labels_ids = None

    assert len(input_ids) == max_len, "Error with input length {} vs {}".format(len(input_ids), max_len)
    assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(len(attention_mask), max_len)
    assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(len(token_type_ids), max_len)

    input_ids = torch.tensor(input_ids).reshape(1,-1).to(device)
    attention_mask = torch.tensor(attention_mask).reshape(1,-1).to(device)
    token_type_ids = torch.tensor(token_type_ids).reshape(1,-1).to(device)
    labels_ids = labels_ids

    model = model.to(device)
    model.eval()
    # 由于传入的tag为None，所以返回的loss 也是None
    ret = model(input_ids = input_ids,
                  tags = labels_ids,
                  attention_mask = attention_mask,
                  token_type_ids = token_type_ids)
    pre_tag = ret[1][0]
    assert len(pre_tag) == len(sentence_list) or len(pre_tag) == max_len - 2

    pre_tag_len = len(pre_tag)
    b_loc_idx = CRF_LABELS.index('B-LOC')
    i_loc_idx = CRF_LABELS.index('I-LOC')
    o_idx = CRF_LABELS.index('O')

    if b_loc_idx not in pre_tag and i_loc_idx not in pre_tag:
        print("没有在句子[{}]中发现实体".format(sentence))
        return ''
    if b_loc_idx in pre_tag:

        entity_start_idx = pre_tag.index(b_loc_idx)
    else:

        entity_start_idx = pre_tag.index(i_loc_idx)
    entity_list = []
    entity_list.append(sentence_list[entity_start_idx])
    for i in range(entity_start_idx+1,pre_tag_len):
        if pre_tag[i] == i_loc_idx:
            entity_list.append(sentence_list[i])
        else:
            break
    return "".join(entity_list)


def semantic_matching(model,tokenizer,question,attribute_list,answer_list,max_length):

    assert len(attribute_list) == len(answer_list)

    pad_token = 0
    pad_token_segment_id = 1
    features = []
    for (ex_index, attribute) in enumerate(attribute_list):
        inputs = tokenizer.encode_plus(
            text = question,
            text_pair = attribute,
            add_special_tokens = True,
            max_length = max_length,
            truncate_first_sequence = True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        features.append(
            SimInputFeatures(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    assert all_input_ids.shape == all_attention_mask.shape
    assert all_attention_mask.shape == all_token_type_ids.shape


    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,batch_size=128)

    data_num = all_attention_mask.shape[0]
    batch_size = 128

    all_logits = None
    for i in range(0,data_num,batch_size):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': all_input_ids[i:i+batch_size].to(device),
                      'attention_mask': all_attention_mask[i:i+batch_size].to(device),
                      'token_type_ids': all_token_type_ids[i:i+batch_size].to(device),
                      'labels': None
                      }
            outputs = model(**inputs)
            logits = outputs[0]
            logits = logits.softmax(dim = -1)

            if all_logits is None:
                all_logits = logits.clone()
            else:
                all_logits = torch.cat([all_logits,logits],dim = 0)
    pre_rest = all_logits.argmax(dim = -1)
    if 0 == pre_rest.sum():
        return torch.tensor(-1)
    else:
        return pre_rest.argmax(dim = -1)


def select_database(sql):
    # connect database
    connect = pymysql.connect(user="root",password="123456",host="127.0.0.1",port=3306,db="kb_qa",charset="utf8")
    cursor = connect.cursor()  # 创建操作游标
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
    except Exception as e:
        print("Error: unable to fecth data: %s ,%s" % (repr(e), sql))
    finally:
        # 关闭数据库连接
        cursor.close()
        connect.close()
    return results


# 文字直接匹配，看看属性的词语在不在句子之中
def text_match(attribute_list,answer_list,sentence):

    assert len(attribute_list) == len(answer_list)

    idx = -1
    for i,attribute in enumerate(attribute_list):
        if attribute in sentence:
            idx = i
            break
    if -1 != idx:
        return attribute_list[idx],answer_list[idx]
    else:
        return "",""



def main():

    with torch.no_grad():
        tokenizer_inputs = ()
        tokenizer_kwards = {'do_lower_case': False,
                            'max_len': 64,
                            'vocab_file': './input/config/bert-base-chinese-vocab.txt'}
        ner_processor = NerProcessor()
        sim_processor = SimProcessor()
        tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)


        ner_model = get_ner_model(config_file = './input/config/bert-base-chinese-config.json',
                                  pre_train_model = './output/best_ner.bin',label_num = len(ner_processor.get_labels()))
        ner_model = ner_model.to(device)
        ner_model.eval()

        sim_model = get_sim_model(config_file='./input/config/bert-base-chinese-config.json',
                                  pre_train_model='./output/best_sim.bin',
                                  label_num=len(sim_processor.get_labels()))

        sim_model = sim_model.to(device)
        sim_model.eval()

        while True:
            print("====="*10)
            raw_text = input("问题：\n")
            raw_text = raw_text.strip()
            if ( "quit" == raw_text ):
                print("quit")
                return
            entity = get_entity(model=ner_model, tokenizer=tokenizer, sentence=raw_text, max_len=64)
            print("实体:", entity)
            if '' == entity:
                print("未发现实体")
                continue
            sql_str = "select * from nlpccqa where entity = '{}'".format(entity)
            triple_list = select_database(sql_str)
            triple_list = list(triple_list)
            if 0 == len(triple_list):
                print("未找到 {} 相关信息".format(entity))
                continue
            triple_list = list(zip(*triple_list))
            # print(triple_list)
            attribute_list = triple_list[1]
            answer_list = triple_list[2]
            attribute, answer = text_match(attribute_list, answer_list, raw_text)
            if attribute != '' and answer != '':
                ret = "{}的{}是{}".format(entity, attribute, answer)
            else:
                sim_model = get_sim_model(config_file='./input/config/bert-base-chinese-config.json',
                                          pre_train_model='./output/best_sim.bin',
                                          label_num=len(sim_processor.get_labels()))

                sim_model = sim_model.to(device)
                sim_model.eval()
                attribute_idx = semantic_matching(sim_model, tokenizer, raw_text, attribute_list, answer_list, 64).item()
                if -1 == attribute_idx:
                    ret = ''
                else:
                    attribute = attribute_list[attribute_idx]
                    answer = answer_list[attribute_idx]
                    ret = "{}的{}是{}".format(entity, attribute, answer)
            if '' == ret:
                print("未找到{}相关信息".format(entity))
            else:
                print("回答:",ret)

if __name__ == '__main__':
    main()

















