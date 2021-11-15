# KgCLUE

KgCLUE: 大规模基于知识图谱的问答


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍背景 |
| [任务描述和统计](#任务描述和统计) | 介绍任务的基本信息 |
| [数据集介绍](#数据集介绍) | 介绍数据集及示例 |
| [实验结果](#实验结果) | 针对各种不同方法，在KgCLUE上的实验对比 |
| [实验分析](#实验分析) | 对人类表现、模型能力和任务进行分析 |
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



## 任务描述

KBQA任务即为给定一份知识库和一份问答数据集，从问答数据集中学习问题语义然后从知识库中查询答案。本测评提供了一份中文百科知识库和一份问答数据集。


## 数据集介绍

### 知识库介绍

知识库可通过<a href="https://pan.baidu.com/s/1NyKw2K5bLEABWtzbbTadPQ">百度云</a>，提取码:nqbq，或着<a href='https://drive.google.com/file/d/1UufUy4_4GK63wmbFnxHu3no7oP_5AOJy/view?usp=sharing'>Google云</a>下载。

知识库基本信息详见[knowledge/README](/knowledge/README.md)。


### 问答数据集介绍

| Corpus   | Train     | Dev  |Test Public| Test Private |
| :----:| :----:  |:----:  |:----:  |:----:  |
| Num Samples |  18k      |   2k   |      2k     |       3k       |
| Num Relations |  2164      |  1258    |     1260      |     423         |

    问答数据集为one-hop数据，总共包含25000条问答对。
    数据集分为4份，1份训练集；1份验证集；1份公开测试集，用于测试；1份私有测试集，用于提交，不公开。



## 实验结果
实验设置：训练集和验证集使用训练集与验证集，测试集使用公开测试集。

| Model   | F1     | EM  |
| :----:| :----:  |:----:  |
| Bert-base-chinese |  81.8      |   79.1   |
| chinese-roberta-wwm-ext-large |  82.6     |  80.6    |

## 实验分析

### 1.人类水平  Human Performance

    

### 2.测评结果  Benchmark Results

#### 2.1 模型评测指标

我们采用业界常用的F1-score 以及完全匹配（Exact Match下简称EM）来作为模型的评测指标

#### 2.2 模型表现分析  Analysis of Model Performance

    两个baseline都使用预训练模型直接做下游任务微调 一个为bert-base-chinese 另一个为chinese-roberta-wwm-ext-large
    我们发现：
    1）参照过往工作，两个模型的F1和EM分数都属于中等水平 说明对于中文KBQA领域，模型还有很大的发展空间
    2）模型的效果会对下游任务分数有所提升

## KgCLUE有什么特点
1、KBQA利用的是结构化的知识，其数据来源决定了适合回答what，when 等事实性问题。

2、KBQA的研究，根据目标问题的性质，可以分为几个方向。第一个是单跳问题 (one hop) ，第二个是多跳问题 (multi hop)。单跳问题是指可以通过知识库中某一条事实三元组来回答，而多跳问题是指需要知识库中的多条事实三元组来回答。KgCLUE第一版针对的是单跳问题构建的数据集。

3、测评的主要目标是KBQA，根据KBQA任务的特点，可以考察近年来的实体识别、关系分类以及实体链接等子任务的发展。

此外，我们提供KBQA测评完善的基础设施。
从任务设定，广泛的数据集，多个有代表性的基线模型及效果对比，一键运行脚本，到测评系统等完整的基础设施。


## 基线模型及运行
    
​    bert-base-chinese 版本

​        环境准备：
​          预先安装Python 3.x, pytorch version >=1.2.0, transformers versoin 2.0。
​          需要预先下载预训练模型：bert-base-chinese，并放入到config目录下（这个文件夹）
        <a href='https://huggingface.co/bert-base-chinese'> bert-base-chinese</a>
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

## 问题 Question
    1. 问：测试系统，什么时候开发？
       答：测评系统在11月15日后才会开放。

## 贡献与参与
    1.问：我有符合代码规范的模型代码，并经过测试，可以贡献到这个项目吗？
     答：可以的。你可以提交一个pull request，并写上说明。
    
    2.问：我正在研究KBQA学习，具有较强的模型研究能力，怎么参与到此项目？
      答：发送邮件到 CLUEbenchmark@163.com，标题为：参与KgCLUE课题，并介绍一下你的研究。


## 引用 Reference

    正在添加中

## License

    正在添加中
## 引用
    {FewCLUE,
      title={FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark},
      author={Liang Xu, Xiaojing Lu, Chenyang Yuan, Xuanwei Zhang, Huilin Xu, Hu Yuan, Guoao Wei, Xiang Pan, Xin Tian, Libo Qin, Hu Hai},
      year={2021},
      howpublished={\url{https://arxiv.org/abs/2107.07498}},
    }
