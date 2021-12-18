## 基准模型baseline(NER_RE)说明

#### Chinese NER using Bert(pytorch) + 相似度模型(bert4keras) +es接口

## 思路介绍

    1.利用NER模型进行实体识别(S)；
    2.根据识别到的实体，通过es接口找到可能的候选关系的列表；
    3.训练相似度模型进行关系预测：输入为问句和候选关系，找到最可能的关系（P）；
    4.最后根据实体（S）、关系(P)定位到答案（O,即尾实体）


## 环境依赖
1）NER模型：
     
     python3.6+
     1.1.0 =< pytorch < 1.5.0, or 1.7.1
    
2）SIM模型

    python3.6+
    tensorflow 1.14+
    bert4keras, 0.10.8

## 效果对比
   <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/ner_re_performance.jpeg"  width="70%" height="70%" /> 

## 目录结构
    prev_trained_model：放预训练模型(pytorch和tensorflow)
    processed_data: 相似度模型的存放的训练数据 
    processors: NER预训练相关的脚本
    scripts: NER训练、预测的shell脚本
    sim: 相似度模型相关，数据生成、训练、预测
    submit:生成测试集上的预测文件
    output:训练模型后模型的checkponts等

## 如何运行

    进入到ner_re的目录(cd baselines/ner_re)，然后循序执行以下1-2-3的命令。

### 1.NER模型(pytorch)
#### 1.0 下载预训练模型
 下载并将预训练模型(<a href='https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD'>chinese_rbt3_pytorch（用于NER）</a>）放入到prev_trained_model目录
 下载并将预训练模型(<a href='https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD'>chinese_rbt3_pytorch（用于NER）</a>）放入到prev_trained_model目录

#### 1.1 训练NER模型
    bash scripts/run_ner_softmax.sh 
    
    其中，处理成NER训练数据的主要代码：processors/utils_ner.py的DataProcessor(65-96行)
   
   已经训练好的NER模型<a href='https://storage.googleapis.com/cluebenchmark/kgclue_models/RBT3_ner.zip'>下载</a>
    
#### 1.2 对测试集(test.json)进行预测，生成NER结果

    bash scripts/run_ner_softmax.sh predict
    
    预测结果在这里：./outputs/kg_output/bert/test_prediction.json
    生成的示例如:
    {"id": 0, "tag_seq": "O O O O B-NER I-NER I-NER O O O O O O", "entities": [["NER", 4, 6]]}
    {"id": 1, "tag_seq": "O O B-NER I-NER I-NER O O O O O O O O", "entities": [["NER", 2, 4]]}
    {"id": 2, "tag_seq": "O O B-NER I-NER I-NER I-NER I-NER I-NER I-NER O O O O O O O O", "entities": [["NER", 2, 8]]}

### 2.SIM（相似度）模型
####  2.1 生成相似度训练数据

    python3 -u  sim/process_sim_data.py
    
    其中，生成的相似度训练数据所在的目录为：./processed_data
   
   已经训练好的相似度(SIM)模型<a href='https://storage.googleapis.com/cluebenchmark/kgclue_models/RBT3_ner.zip'>下载</a>

#### 2.2 训练SIM模型

    python3 -u sim/train.py
    
    其中，相似度模型所在的位置：./outputs/kg_sim_output
    
#### 测试单个输入的相似度（可选）

    python3 -u sim/predict.py

### 3. 生成预测文件并提交

    python3 -u submit/generate_submit_file.py
    
    生成的文件为：./kgclue_predict_rbt3.json

    使用如下命令压缩文件：zip -r kgclue_predict_rbt3.zip kgclue_predict_rbt3.json
    
   提交预测文件到<a href='www.CLUEbenchmarks.com'>测评系统</a>，并查看：<a href='https://www.cluebenchmarks.com/kgclue.html'>榜单效果</a>

### 根据实体获得候选关系(-答案-实体释义)的es接口
    
    import requests, json
    data = {'entity': '马云'}
    url = 'http://47.75.32.69:5004/search_kb/entity/'
    r = requests.post(url, data=json.dumps(data))
    # print("r:",r.text)
    data=json.loads(r.text)["data"] # ["attribute_list"]
    print("data:",data)
    
    测试见：relationshp_by_entity_test.py
    
## 如何进一步提升效果？
1）NER模型有部分实体没能识别出来，定位到问题，并设法缓解。
2）有一定比例的问题没能得到答案，分析一下中间过程，找到可能的原因。


### 支持的NER模型或方法类型 

1. BERT+Softmax
2. BERT+CRF
3. BERT+Span


### 参考项目或原始代码来源
<a href='https://github.com/lonePatient/BERT-NER-Pytorch'>BERT-NER-Pytorch(NER模型的代码)</a>
<a href='https://github.com/bojone/bert4keras/blob/master/examples/task_sentence_similarity_lcqmc.py'>bert4keras(相似度模型的代码)</a>
<a href='https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD'>Chinese-BERT-wwm(预训练模型)</a>