import csv
import json
import torch
import numpy as np
from transformers import BertTokenizer

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self,input_file):
        lines = []
        with open(input_file,'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words":words,"labels":labels})
        return lines

    @classmethod
    def _read_json(self,input_file):
        lines = []
        count_error=0
        count_total=0
        with open(input_file,'r') as f:
            # line: {"id": 19, "question": "济河的发源地是哪里？", "answer": "济河（豫鲁的河流） ||| 发源于 ||| 河南省济源市"}
            for indexx,line in enumerate(f):
                line = json.loads(line.strip())
                count_total=count_total+1
                question = line['question']
                answer = line.get('answer',None)
                words = list(question)
                labels = ['O'] * len(words)
                if answer is None: # 如果answer为空，那么是预测阶段；不用额外处理标签(label)数据了
                    lines.append({"words": words, "labels": labels})
                    continue
                entity=answer.split("|||")[0].strip()
                if "（" in entity:
                    entity=entity.split("（")[0]
                # print("question:",question,";answer:",answer,";entity:",entity)
                start_index=-1
                if entity in question:
                    start_index=question.index(entity)
                else: # 如果正常方式找不到，那么做一下处理，然后将公共子串作为实体。
                    # 1)移除答案(answer)中的“·”、空格；2)取最长公共子串作为实体(entity)
                    print("ERROR1.processors.utils_ner.read_json. question:"+question+";answer:",answer+";entity:"+entity)
                    entity=entity.replace("·","").replace(" ","").strip()
                    # 找到最长公共子串
                    common_str_list=find_longest_common_str(question, entity)
                    if common_str_list is None: # 如果没有公共子串的跳过
                        count_error = count_error + 1
                        continue
                    target_common_str=''
                    max_common_str_length=-1
                    for common_str in common_str_list:
                        if len(common_str)>max_common_str_length:
                            max_common_str_length=len(common_str)
                            target_common_str=common_str
                    # 将最长公共子串作为目标实体
                    entity=target_common_str
                    print("CORRECT2.processors.utils_ner.read_json. question:"+question+";answer:",answer+";entity:"+entity)
                length_entity=len(entity)
                for ind in range(len(words)):
                    if ind>=start_index and ind<start_index+length_entity:
                        if length_entity==1: # 单个的
                            labels[ind] = "S-NER"
                        elif ind==start_index: # 开始
                            labels[ind]="B-NER"
                        elif ind<=start_index+length_entity-1: # 中间
                            labels[ind] = "I-NER"
                        #else:
                        #    labels[ind] = "S-NER"

                if indexx<5:
                    print("words:",words,";labels:",labels,";answer:",answer)
                lines.append({"words": words, "labels": labels})
        print("utis_ner._read_json.count_error:",count_error,";count_total:",count_total)
        # if count_error>0:
        #     iii=0;iii/0
        return lines

def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)

def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    # print("utils_ner.bert_extract_item.start_logits:",start_logits,";start_pred:",start_pred)
    # print("utils_ner.bert_extract_item.end_logits:",end_logits,";end_pred:",end_pred) # end_pred: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    if np.sum(end_pred)==0:
        numpy_array=np.array(end_logits[0][:, 1].cpu())[1:-1]
        end_index=np.argmax(numpy_array)
        end_pred[end_index]=1
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S



def find_longest_common_str(s1,s2):
    """
    查找最长公共子串
    :param s1:
    :param s2:
    :return:
    """
    if len(s1) > len(s2):
        s1,s2 = s2,s1
        print(s1,s2)
    length = len(s1)
    result = []
    for step in range(length,0 ,-1):
        for start in range(0,length-step+1):
            flag = True
            tmp = s1[start:start+step]
            if s2.find(tmp)>-1 :# 第一次找到,后面要接着找
                result.append(tmp)
                flag = True
                newstart = start+1
                newstep = step
                while flag:   # 已经找到最长子串,接下来就是判断后面是否还有相同长度的字符串
                    if newstart+ step >length:  # 大于字符串总长了,退出循环
                        break
                    newtmp = s1[newstart:newstart+newstep]
                    if s2.find(newtmp)>-1:
                        result.append(newtmp)
                        newstart+=1
                        flag = True
                    else:
                        newstart +=1
                        flag = True
                return result
            else:
                continue


# str1 = 'abcdefgg好好学习denf'
# str2 = 'denf好好学习abcd'
# result=str_int(str1,str2)
# print("result:",result)