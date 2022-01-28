# KgCLUE

KgCLUE: 大规模中文开源知识图谱问答

在线DEMO<a href="https://www.cluebenchmarks.com/KgCLUEdemo">点击此处</a>体验


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍背景 |
| [任务描述](#任务描述) | 介绍任务的基本信息 |
| [数据集介绍](#数据集介绍) | 介绍数据集及示例 |
| [实现思路](#实现思路) | 介绍实现的具体思路 |
| [实验结果](#实验结果) | 针对各种不同方法，在KgCLUE上的实验对比 |
| [实验分析](#实验分析) | 对模型能力进行分析 |
| [KgCLUE有什么特点](#KgCLUE有什么特点) | 特点介绍 |
| [基线模型及运行](#基线模型及运行) | 支持多种基线模型 |
| [相关阅读](#相关阅读) | 新方案及其解读 |
| [排行榜及提交](#排行榜及提交) | 排行榜及提交样例 |
| [贡献与参与](#贡献与参与) | 如何参与项目或反馈问题|

  <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/kgclue1.0.gif"  width="75%" height="75%" />   


## 简介
 KBQA（Knowledge Base Question Answering），即给定自然语言问题，通过对问题进行语义理解和解析，进而利用知识库进行查询、推理得出答案。
 
 KBQA利可以用图谱丰富的语义关联信息，能够深入理解用户问题并给出答案，近年来吸引了学术界和工业界的广泛关注。KBQA主要任务是将自然语言问题（NLQ）通过不同方法映射到结构化的查询，并在知识图谱中获取答案。
 
 KgCLUE：中文KBQA测评基准，基于CLUE的积累和经验，并结合KBQA的特点和近期的发展趋势，精心设计了该测评，希望可以促进中文领域上KBQA领域更多的研究、应用和发展。


### UPDATE:
  ******* 2021-12-18：添加完整基线模型(baselines/ner_sim)，包括预训练脚本及训练好的模型的下载地址
   
  ******* 2021-11-27：添加支持KgCLUE的Roberta-wwm-ext的baseline

  ******* 2021-11-15：添加支持KgCLUE的Roberta-wwm-large的baseline

  ******* 2021-11-03: 添加支持KgCLUE的Bert的baseline

  ******* 2021-11-02: 添加了知识库和问答数据集。
  

## 任务描述

KBQA任务即为给定一份知识库和一份问答数据集，从问答数据集中学习问题语义然后从知识库中查询答案。本测评提供了一份中文百科知识库和一份问答数据集。


## 数据集介绍

### 知识库介绍

知识库可通过<a href="https://pan.baidu.com/s/1bJgDGz0NjU1EtMjBWjQayw">百度云</a>，提取码:nhsb，或着<a href='https://drive.google.com/file/d/1tOSwVzr71uJjHbMZJd67yT8kEAc5ThqL/view?usp=sharing'>Google云</a>下载。

下载后，请将其放入knowledge文件夹中。

### 知识库（三元组）统计信息
| 实体数量   | 关系数量     | 高频关系(>100)  |三元组数量| 
| :----:| :----:  |:----:  |:----:  | 
|   3121457    |    245838     |   3833     |   20559652    |

  知识库来源于百科类数据，由百科类搜索页面的事实性三元组构成。


### 知识库描述

  <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/knowledge_info.png"  width="85%" height="85%" />   
  <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/kg_example2.jpeg"  width="95%" height="95%" />   


    知识库中数据存储格式如上，每一行是一个三元组，格式为<头实体，关系，尾实体>，每列之间以'\t'分隔，
    其中头实体后的括号项为该实体的消歧项。


### 问答数据集统计信息

| Corpus   | Train     | Dev  |Test Public| Test Private |
| :----:| :----:  |:----:  |:----:  |:----:  |
| Num Samples |  18k      |   2k   |      2k     |       3k       |
| Num Relations |  2164      |  1258    |     1260      |     423         |

    问答数据集为one-hop数据，总共包含25000条问答对。
    数据集分为4份：1份训练集(Train)；1份验证集(Dev)；1份公开测试集(Test Public)，用于测试；1份私有测试集(Test Private)，用于提交，不公开。

### 问答数据集描述

   <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/qa.png"  width="100%" height="100%" />  

问答数据集为json格式，每行为一条问答对。问题是one-hop问题，即答案为知识库中的一条三元组。数据格式如下，其中id为问答对索引，quetion为问题，answer为答案，来自知识库，以' ||| '分割。


## 实验结果
实验设置：训练集和验证集使用训练集与验证集，测试集使用公开测试集。

   <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/ner_re_performance.jpeg"  width="70%" height="70%" /> 


## 实现思路
下面简单介绍该任务的baseline的构建思路，但并不对任务数据进行详细介绍，如对任务数据不明白，请返回[数据集介绍](#数据集介绍)部分

以下图片仅为思路参考，总体思路：

    1.利用NER模型进行实体识别(S)；
    2.根据识别到的实体，通过es接口找到可能的候选关系的列表；
    3.训练相似度模型进行关系预测：输入为问句和候选关系，找到最可能的关系（P）；
    4.最后根据实体（S）、关系(P)定位到答案（O,即尾实体）

#### NER阶段
 做NER任务，如图所示用于识别出问题中的实体。
 
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForTokenClassification+crf.png"  width="100%" height="100%" /> 

<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/Bert-Bilstm-CRF.png"  width="100%" height="100%" /> 

#### SIM阶段 
我们使用句子分类任务（二分类），把问题和关系(属性）拼接到一起，用于判断问题要问的是不是这个属性。

<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForSequenceClassification.png"  width="85%" height="85%" /> 

## 基线模型及运行 
### 环境依赖
1）NER模型：   
     python3.6+
     1.1.0 =< pytorch < 1.5.0, or 1.7.1
2）SIM模型
    python3.6+
    tensorflow 1.14+
    bert4keras, 0.10.8 
### 如何运行
    进入到ner_re的目录(cd baselines/ner_re)；
    然后顺序执行以下命令：1）训练NER模型；2）训练相似度模型；3）生成预测文件并提交。
#### 1.NER模型(pytorch)
##### 1.0 下载预训练模型
 下载并将预训练模型(<a href='https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD'>chinese_rbt3_pytorch</a>）放入到prev_trained_model目录。
 pytorch版用于NER,tensorflow版用于相似度模型。

##### 1.1 训练NER模型
    bash scripts/run_ner_softmax.sh 
    
    其中，处理成NER训练数据的主要代码：processors/utils_ner.py的DataProcessor(65-96行)
   
   已经训练好的NER模型<a href='https://storage.googleapis.com/cluebenchmark/kgclue_models/RBT3_ner.zip'>下载</a>
    
##### 1.2 对测试集(test.json)进行预测，生成NER结果

    bash scripts/run_ner_softmax.sh predict
    
    预测结果在这里：./outputs/kg_output/bert/test_prediction.json
    生成的示例如:
    {"id": 0, "tag_seq": "O O O O B-NER I-NER I-NER O O O O O O", "entities": [["NER", 4, 6]]}
    {"id": 1, "tag_seq": "O O B-NER I-NER I-NER O O O O O O O O", "entities": [["NER", 2, 4]]}
    {"id": 2, "tag_seq": "O O B-NER I-NER I-NER I-NER I-NER I-NER I-NER O O O O O O O O", "entities": [["NER", 2, 8]]}
#### 2.SIM（相似度）模型
#####  2.1 生成相似度训练数据

    python3 -u  sim/process_sim_data.py
    
    其中，生成的相似度训练数据所在的目录为：./processed_data
   
   已经训练好的相似度(SIM)模型<a href='https://storage.googleapis.com/cluebenchmark/kgclue_models/RBT3_ner.zip'>下载</a>

##### 2.2 训练SIM模型

    python3 -u sim/train.py
    
    其中，相似度模型所在的位置：./outputs/kg_sim_output
    
##### 测试单个输入的相似度（可选）

    python3 -u sim/predict.py
#### 3. 生成预测文件并提交

    python3 -u submit/generate_submit_file.py
    
    生成的文件为：./kgclue_predict_rbt3.json

    使用如下命令压缩文件：zip -r kgclue_predict_rbt3.zip kgclue_predict_rbt3.json
    
   提交预测文件到<a href='www.CLUEbenchmarks.com'>测评系统</a>，并查看：<a href='https://www.cluebenchmarks.com/kgclue.html'>榜单效果</a>

基线模型详细介绍见：<a href='./baselines/ner_re/README.md'>./baselines/ner_re/README.md</a>


### 相关阅读
   <a href='https://kexue.fm/archives/8802'>《Seq2Seq+前缀树：检索任务新范式（以KgCLUE为例）》，苏剑林</a>
    
    本文介绍了检索模型的一种新方案——“Seq2Seq+前缀树”，并以KgCLUE为例给出了一个具体的baseline。
    “Seq2Seq+前缀树”的方案有着训练简单、空间占用少等优点，也有一些不足之处，总的来说算得上是一种简明的有竞争力的方案。
    
>我们还提供了另一个代码库可以更简单方便的复现我们的效果https://github.com/CLUEbenchmark/KgCLUEbench

## 效果评估脚本
     Score=EM_O * 0.50 + F1_O * 0.50

<a href='./baselines/evaluate_f1_em.py'>evaluate_f1_em.py</a>

## 实验分析


### 1.测评结果  Benchmark Results

#### 1.1 模型评测指标

我们采用业界常用的F1-score 以及完全匹配（Exact Match下简称EM）来作为模型的评测指标

F1-score：F1-score是分类问题的常用指标，广泛用于 QA。 当我们同样关心精度和召回率时用F1就十分合适。 在这种情况下，它是针对预测中的单个单词与真实答案中的单词进行计算的。 预测和真值之间的共享词数是F1分数的基础：精度是预测中共享词的数量与总词数的比值，召回率是共享词数的比值 到基本事实中的单词总数。
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/f1score.png"  width="85%" height="85%" /> 

完全匹配(Exact Match)：对于每个问题+答案对，如果模型预测的答案的字符与正确答案（之一）的字符完全匹配，则 EM = 1，否则 EM = 0。这是一个严格的有或无的指标； 如有单字错误仍然得分为 0。在针对负面示例进行评估时，如果模型预测了任何文本，它会自动为该示例得分为 0。

#### 1.2 模型表现分析  Analysis of Model Performance

baseline都使用预训练模型直接做下游任务微调 bert-base-chinese，chinese-roberta-wwm-ext-large以及chinese-roberta-wwm-ext

我们发现：

1）参照过往工作，三个个模型的F1和EM分数都属于中等水平 说明对于中文KBQA领域，模型还有很大的发展空间

2）模型的效果会对下游任务分数有所提升

3）在NER和similarity的阶段的效果影响结果较大，参考过往工作，我们的baseline模型两阶段分数处于中下等水平，还是有很大的发展空间 

4）我们曾使用过不同难度的数据训练以及测试模型，发现数据处理对分数的影响较大，或许可以设法通过难样本挖掘构建难样本进行更有效训练


## KgCLUE有什么特点
1、KBQA利用的是结构化的知识，其数据来源决定了适合回答what，when 等事实性问题。

2、KBQA的研究，根据目标问题的性质，可以分为几个方向。第一个是单跳问题 (one hop) ，第二个是多跳问题 (multi hop)。单跳问题是指可以通过知识库中某一条事实三元组来回答，而多跳问题是指需要知识库中的多条事实三元组来回答。KgCLUE第一版（即KgCLUE1.0）针对的是单跳问题构建的数据集。

3、测评的主要目标是KBQA，根据KBQA任务的特点，可以考察近年来的实体识别、关系分类以及实体链接等子任务的发展。

此外，我们提供KBQA测评完善的基础设施。
从任务设定，广泛的数据集，多个有代表性的基线模型及效果对比，一键运行脚本，到测评系统等完整的基础设施。

## 排行榜及提交

#### 提交说明
训练端到端或非端到端模型，在非公开测试集上<a href="./qa_data/test.json">test.json</a>进行预测，
生成kgclue_predict.json并压缩，得到kgclue_predict.zip；然后提交到<a href="https://www.cluebenchmarks.com" target="_blank">CLUE测评系统</a>

<a href="./resources/kgclue_submit_examples/kgclue_predict.json">提交样例</a>

<a href="https://www.cluebenchmarks.com/kgclue.html" target="_blank">排行榜</a>


## 问题 Question
    1. 问：测试系统，什么时候开发？
       答：测评系统在2021年12月1日后开放。
    2. 问：SIM训练的数据集标注怎么搞？
       答：问题原样本属性为正，再随机从样本中其他属性抽5个设为负（你也可以设法通过难样本挖掘构建难样本进行更有效训练）。
    3. 问：什么是属性？
       答：三元组中间那列数据，如图所示
  <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/attribute.png"  width="85%" height="85%" /> 


## 贡献与参与
    1.问：我有符合代码规范的模型代码，并经过测试，可以贡献到这个项目吗？
     答：可以的。你可以提交一个pull request，并写上说明。我们会设法在24小时内反馈。
    
    2.问：我正在研究KBQA学习，具有较强的模型研究能力，怎么参与到此项目？
      答：发送邮件到 CLUEbenchmark@163.com，标题为：参与KgCLUE课题，并介绍一下你的研究。
    
    3.如何交流？
     提交你的issue；加QQ群（群号:836811304）；或加入微信群
   <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/kgcluegroup.jpeg"  width="35%" height="35%" /> 
   <img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/brightmart.jpeg"  width="35%" height="35%" /> 



