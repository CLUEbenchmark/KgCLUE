# KgCLUE

KgCLUE: 大规模基于知识图谱的问答

我们制作了一个简单的demo，请<a href="http://www.cluebenchmarks.com:5000">点击此处</a>体验


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍背景 |
| [任务描述和统计](#任务描述和统计) | 介绍任务的基本信息 |
| [数据集介绍](#数据集介绍) | 介绍数据集及示例 |
| [实验结果](#实验结果) | 针对各种不同方法，在KgCLUE上的实验对比 |
| [实验分析](#实验分析) | 对模型能力进行分析 |
| [KgCLUE有什么特点](#KgCLUE有什么特点) | 特点介绍 |
| [基线模型及运行](#基线模型及运行) | 支持多种基线模型 |

| [贡献与参与](#贡献与参与) | 如何参与项目或反馈问题|


## 简介
 KBQA（Knowledge Base Question Answering），即给定自然语言问题，通过对问题进行语义理解和解析，进而利用知识库进行查询、推理得出答案。
 
 KBQA利可以用图谱丰富的语义关联信息，能够深入理解用户问题并给出答案，近年来吸引了学术界和工业界的广泛关注。KBQA主要任务是将自然语言问题（NLQ）通过不同方法映射到结构化的查询，并在知识图谱中获取答案。
 
 KgCLUE：中文KBQA测评基准，基于CLUE的积累和经验，并结合KBQA的特点和近期的发展趋势，精心设计了该测评，希望可以促进中文领域上KBQA领域更多的研究、应用和发展。


### UPDATE:
  
  ******* 2021-11-02: 添加了知识库和问答数据集。
  
  ******* 2021-11-03: 添加支持KgCLUE的Bert的baseline

  ******* 2021-11-15：添加支持KgCLUE的Roberta-wwm-large的baseline

  ******* 2021-11-27：添加支持KgCLUE的Roberta-wwm-ext的baseline



## 任务描述

KBQA任务即为给定一份知识库和一份问答数据集，从问答数据集中学习问题语义然后从知识库中查询答案。本测评提供了一份中文百科知识库和一份问答数据集。


## 数据集介绍

### 知识库介绍

知识库可通过<a href="https://pan.baidu.com/s/1NyKw2K5bLEABWtzbbTadPQ">百度云</a>，提取码:nqbq，或着<a href='https://drive.google.com/file/d/1UufUy4_4GK63wmbFnxHu3no7oP_5AOJy/view?usp=sharing'>Google云</a>下载。

下载后，请将其放入knowledge文件夹中。

### 知识库统计信息
| 实体数量   | 关系数量     | 高频关系(>100)  |三元组数量| 
| :----:| :----:  |:----:  |:----:  | 
|   3137356    |    246380     |   4143     |   23022248     |

  知识库来源于百科数据，由百科搜索页面的事实性三元组构成。


### 知识库描述

  <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/knowledge_info.png"  width="100%" height="100%" />   


    知识库中数据存储格式如上，每一行是一个三元组，格式为<头实体，关系，尾实体>，每列之间以'\t'分隔，其中头实体后的括号项为该实体的消歧项。


### 问答数据集统计信息

| Corpus   | Train     | Dev  |Test Public| Test Private |
| :----:| :----:  |:----:  |:----:  |:----:  |
| Num Samples |  18k      |   2k   |      2k     |       3k       |
| Num Relations |  2164      |  1258    |     1260      |     423         |

    问答数据集为one-hop数据，总共包含25000条问答对。
    数据集分为4份，1份训练集；1份验证集；1份公开测试集，用于测试；1份私有测试集，用于提交，不公开。

### 问答数据集描述

   <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/qa.png"  width="100%" height="100%" />  

问答数据集为json格式，每行为一条问答对，问题是one-hop问题，即答案为知识库中的一条三元组。数据格式如下，其中id为问答对索引，quetion为问题，answer为答案，来自知识库，以' ||| '分割。


## 实验结果
实验设置：训练集和验证集使用训练集与验证集，测试集使用公开测试集。

| Model   | F1     | EM  |
| :----:| :----:  |:----:  |
| Bert-base-chinese |  81.8      |   79.1   |
| chinese-roberta-wwm-ext-large |  82.6     |  80.6    |

| Model   | NER（F1-score）    | SIM(F1-score)  |
| :----:| :----:  |:----:  |
| Bert-base-chinese |  76.1      |   74.8   |
| chinese-roberta-wwm-ext-large |  81.3     |  76.93    |
| chinese-roberta-wwm-ext |  84.1     |  86.2   |


## 实验分析


### 1.测评结果  Benchmark Results

#### 1.1 模型评测指标

我们采用业界常用的F1-score 以及完全匹配（Exact Match下简称EM）来作为模型的评测指标

F1-score：F1-score是分类问题的常用指标，广泛用于 QA。 当我们同样关心精度和召回率时用F1就十分合适。 在这种情况下，它是针对预测中的单个单词与真实答案中的单词进行计算的。 预测和真值之间的共享词数是F1分数的基础：精度是预测中共享词的数量与总词数的比值，召回率是共享词数的比值 到基本事实中的单词总数。
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/f1-score.png"  width="100%" height="100%" /> 

完全匹配(Exact Match)：对于每个问题+答案对，如果模型预测的答案的字符与正确答案（之一）的字符完全匹配，则 EM = 1，否则 EM = 0。这是一个严格的有或无的指标； 如有单字错误仍然得分为 0。在针对负面示例进行评估时，如果模型预测了任何文本，它会自动为该示例得分为 0。

#### 1.2 模型表现分析  Analysis of Model Performance

两个baseline都使用预训练模型直接做下游任务微调 一个为bert-base-chinese 另一个为chinese-roberta-wwm-ext-large
我们发现：
1）参照过往工作，两个模型的F1和EM分数都属于中等水平 说明对于中文KBQA领域，模型还有很大的发展空间
2）模型的效果会对下游任务分数有所提升
3) 在NER和similarity的阶段的效果影响结果较大，参考过往工作，我们的baseline模型两阶段分数处于中下等水平，还是有很大的发展空间 

## KgCLUE有什么特点
1、KBQA利用的是结构化的知识，其数据来源决定了适合回答what，when 等事实性问题。

2、KBQA的研究，根据目标问题的性质，可以分为几个方向。第一个是单跳问题 (one hop) ，第二个是多跳问题 (multi hop)。单跳问题是指可以通过知识库中某一条事实三元组来回答，而多跳问题是指需要知识库中的多条事实三元组来回答。KgCLUE第一版针对的是单跳问题构建的数据集。

3、测评的主要目标是KBQA，根据KBQA任务的特点，可以考察近年来的实体识别、关系分类以及实体链接等子任务的发展。

此外，我们提供KBQA测评完善的基础设施。
从任务设定，广泛的数据集，多个有代表性的基线模型及效果对比，一键运行脚本，到测评系统等完整的基础设施。

## 代码简介
下面简单接受下该任务的baseline的构建思路，但并不对任务数据进行详细介绍，如对任务数据不明白，请返回[数据集介绍](#数据集介绍)部分

bert-base-chinese版本

NER阶段： 我们使用BertForTokenClassification + crf来做NER任务，如图所示用于识别出问题中的实体
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForTokenClassification+crf.png"  width="100%" height="100%" /> 
SIM阶段：我们使用BertForSequenceClassification，用于句子分类。把问题和属性拼接到一起，用于判断问题要问的是不是这个属性
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForSequenceClassification.png"  width="100%" height="100%" /> 

chinese-roberta-wwm-ext-large模型：
我们同样适用NER+SIM的思路
但在NER阶段采用了BERT+LSTM+CRF用于识别问题中的实体。输入是wordPiece tokenizer得到的tokenid，进入Bert预训练模型抽取丰富的文本特征得到batch_size * max_seq_len * emb_size的输出向量，输出向量过Bi-LSTM从中提取实体识别所需的特征，得到batch_size * max_seq_len * (2*hidden_size)的向量，最终进入CRF层进行解码，计算最优的标注序列。
如图所示
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/Bert-Bilstm-CRF.png"  width="100%" height="100%" /> 

SIM阶段采用BERT作二分类模型


## 基线模型及运行
数据处理
```：
1、运行baseline/dataProcessing/NERdata.py 处理NER数据
2、运行baseline/dataProcessing/SIMdata.py 处理SIM数据
```   
​    bert-base-chinese模型：

​        环境准备：
​          预先安装Python 3.x, pytorch version >=1.2.0, transformers versoin 2.0。
​          需要预先下载预训练模型：bert-base-chinese，并放入到config目录下（这个文件夹）
        <a href='https://huggingface.co/bert-base-chinese'> bert-base-chinese</a>
          需要将第一步数据处理好的数据放入input文件夹
​        
​        运行：

​        1、进入到相应的目录，运行相应的代码：
```
​           cd ./baseline/bert
```
​        2、运行代码
```
​           python3 NER.py 训练NER的模型
           python3 SIM.py 训练相似度的模型
           python3 test_kbqa.py 测试kbqa
```

chinese-roberta-wwm-ext-large模型：
```
环境准备：
    预先安装Python 3.x(或2.7), Tesorflow 1.14+, Keras 2.3.1, bert4keras。
    需要预先下载预训练模型：chinese_roberta_wwm_ext-large，并放入到ModelParams目录下
    需要将预先处理好的数据放入data文件夹中
```
运行：

1、进入到相对应的目录：
```
cd ./baseline/RoBERTa-wwm-large
```
2、运行代码
```
./run_ner.sh
./terminal_ner.sh
python3 args.py
python3 run_similarity.py
python3 kbqa_test.py
```


chinese-roberta-wwm-ext模型：
```
环境准备：
    预先安装Python 3.x(或2.7), Tesorflow 1.14+
    需要预先下载预训练模型：chinese_roberta_wwm_ext_L-12_H-768_A-12，并放入到chinese_roberta_wwm_ext_L-12_H-768_A-12目录下
```
运行：

1、进入到相对应的目录：
```
cd ./baseline/RoBERTa-wwm-ext
```
2、运行代码
```
实体识别数据预处理
修改好输入与输出的位置后运行 ./NER/ner_data_making.py
训练实体识别
./NER/ner_train.py
文本相似度分类据预处理
修改好输入与输出的位置后运行 ./SIM/sim_data_making.py
训练文本相似度分类
./SIM/sim_train.py
```
## 问题 Question
    1. 问：测试系统，什么时候开发？
       答：测评系统在11月15日后才会开放。
    2. 问：SIM训练的数据集标注怎么搞？
       答：问题原样本属性为正，再随机从样本中其他属性抽5个设为负。
    3. 问：什么是属性？
       答：三元组中间那列数据，如图所示
  <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/attribute.png"  width="100%" height="100%" /> 


## 贡献与参与
    1.问：我有符合代码规范的模型代码，并经过测试，可以贡献到这个项目吗？
     答：可以的。你可以提交一个pull request，并写上说明。
    
    2.问：我正在研究KBQA学习，具有较强的模型研究能力，怎么参与到此项目？
      答：发送邮件到 CLUEbenchmark@163.com，标题为：参与KgCLUE课题，并介绍一下你的研究。


