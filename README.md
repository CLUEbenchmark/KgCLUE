# KgCLUE

KgCLUE: 大规模基于知识图谱的问答


<a href='https://arxiv.org/abs/2107.07498'>kgCLUE: A Large-scale Knowledge Graph Question Answering Evaluation Benchmark for Chinese</a>


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍背景 |
| [任务描述和统计](#任务描述和统计) | 介绍任务的基本信息 |
| [数据集介绍](#数据集介绍) | 介绍数据集及示例 |
| [实验结果](#实验结果) | 针对各种不同方法，在KgCLUE上的实验对比 |
| [实验分析](#实验分析) | 对人类表现、模型能力和任务进行分析 |
| [KgCLUE有什么特点](#KgCLUE有什么特点) | 特定介绍 |
| [基线模型及运行](#基线模型及运行) | 支持多种基线模型 |

| [贡献与参与](#贡献与参与) | 如何参与项目或反馈问题|


## 简介
 KBQA（Knowledge Base Question Answering），即给定自然语言问题，通过对问题进行语义理解和解析，进而利用知识库进行查询、推理得出答案。
 
 KBQA利可以用图谱丰富的语义关联信息，能够深入理解用户问题并给出答案，近年来吸引了学术界和工业界的广泛关注。KBQA主要任务是将自然语言问题（NLQ）通过不同方法映射到结构化的查询，并在知识图谱中获取答案。
 
 KgCLUE：中文KBQA测评基准，基于CLUE的积累和经验，并结合KBQA的特点和近期的发展趋势，精心设计了该测评，希望可以促进中文领域上KBQA领域更多的研究、应用和发展。
  

### UPDATE:
  
  ******* 2021-11-02: 添加了知识库和问答数据集。
  
  ******* 2021-11-03: 添加支持KgCLUE的Bert的baseline


## 任务描述

KBQA任务即为给定一份。本测评提供了一份中文百科知识库和一份问答数据集。


## 数据集介绍

### 知识库介绍

知识库可通过<a href="https://pan.baidu.com/s/1NyKw2K5bLEABWtzbbTadPQ">百度云</a>，提取码:nqbq，或着<a href='https://arxiv.org/abs/2107.07498'>Google云</a>下载。

知识库基本信息详见[knowledge/README](/knowledge/README.md)。


### 问答数据集介绍

| Corpus   | Train     | Dev  |Test Public| Test Private |
| :----:| :----:  |:----:  |:----:  |:----:  |
| Num Samples |        |      |           |              |
| Num Relations |        |      |           |              |

问答数据集为one-hop数据，总共包含25000条问答对，切分为4份数据集。


## 实验结果
实验设置：训练集和验证集使用32个样本，或采样16个，测试集正常规模。基础模型使用RoBERT12层chinese_roberta_wwm_ext（GPT系列除外）。

   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/fewclue_eresult.jpeg"  width="100%" height="100%" />   

    Human: 人类测评成绩；FineTuning: 直接下游任务微调；PET:Pattern Exploiting Training(完形填空形式); 
    Ptuning: 自动构建模板; Zero-shot: 零样本学习；EFL:自然语言推理形式; ADAPET:PET改进版，带正确标签条件优化
    FineTuningB:FineTuningBert; FineTuningR:FineTuningRoberta; PtuningB:Ptuning_RoBERTa; PtuningGPT:Ptuning_GPT; 
    Zero-shot-R，采用chinese_roberta_wwm_ext为基础模型的零样本学习；Zero-shot-G，GPT系列的零样本学习；N”，代表已更新；
    报告的数字是每一个任务的公开测试集(test_public.json)上的实验效果；CLUE榜单已经可以提交；
    由于CHID还在继续实验中，暂时未将CHID的分数纳入到最终的分数(Score）中。

## 实验分析

### 1.人类水平  Human Performance

    我们采取如下方式测评人类水平。按照训练，然后测评的方式。首先，人类会在训练集上学习30个样本，
    然后我们鼓励人类标注员进行讨论和分享经验；然后，每一个人人类标注员在验证集上标注100个样本；最后，由多个人的投票得出最终预测的答案。
    
    从实验结果可以看到，人类有高达82.49分的成绩。人类在多个任务中可以取得超过80分以上的分数。在较难的指代消解任务中，人类甚至达到了高达98的分数；
    而在类别特别多的任务，如iflytek（119个类别）,csldcp(67个类别），人类仅取得了60多分的及格成绩。

### 2.测评结果  Benchmark Results

#### 2.1 模型表现分析  Analysis of Model Performance

    模型有5种不同的方式做任务，分别是使用预训练模型直接做下游任务微调、PET、RoBERTa为基础的Ptuning方式、GPT类模型为基础的Ptuning方式、
    使用RoBERTa或GPT做零样本学习。
    
    我们发现：
    1）模型潜力：最好的模型表现（54.34分）远低于人类的表现（82.49分），即比人类低了近30分。说明针对小样本场景，模型还有很大的潜力；
    2）新的学习范式：在小样本场景，新的学习方式（PET,Ptuning）的效果以较为明显的差距超过了直接调优的效果。
       如在通用的基础模型（RoBERTa）下，PET方式的学习比直接下游任务微调高了近8个点。
    3）零样本学习能力：在没有任何数据训练的情况下，零样本学习在有些任务上取得了较好的效果。如在119个类别的分类任务中，模型在没有训练的情况下
    取得了27.7的分数，与直接下游任务微调仅相差2分，而随机猜测的话仅会获得1%左右的准确率。这种想象在在67个类别的分类任务csldcp中也有表现。

#### 2.2 任务分析  Analysis of Tasks 
    我们发现，在小样本学习场景：
    不同任务对于人类和模型的难易程度相差较大。如wsc指代消解任务，对于人类非常容易（98分），但对于模型却非常困难（50分左右），只是随机猜测水平；
    而有些任务对于人类比较困难，但对于模型却不一定那么难。如csldcp有67个类别，人类只取得了及格的水平，但我们的基线模型PET在初步的实验中
    就取得了56.9的成绩。我们可以预见，模型还有不少进步能力。

## KgCLUE有什么特点
1、任务类型多样、具有广泛代表性。包含多个不同类型的任务，包括情感分析任务、自然语言推理、多种文本分类、文本匹配任务和成语阅读理解等。

2、研究性与应用性结合。在任务构建、数据采样阶段，即考虑到了学术研究的需要，也兼顾到实际业务场景对小样本学习的迫切需求。
如针对小样本学习中不实验结果的不稳定问题，我们采样生成了多份训练和验证集；考虑到实际业务场景类别，我们采用了多个有众多类别的任务（如50+、100+多分类），
并在部分任务中存在类别不均衡的问题。

3、时代感强。测评的主要目标是考察小样本学习，我们也同时测评了模型的零样本学习、半监督学习的能力。不仅能考察BERT类擅长语言理解的模型，
也可以同时查考了近年来发展迅速的GPT-3这类生成模型的中文版本在零样本学习、小样本学习上的能力；

此外，我们提供小样本测评完善的基础设施。
从任务设定，广泛的数据集，多个有代表性的基线模型及效果对比，一键运行脚本，小样本学习教程，到测评系统、学术论文等完整的基础设施。


## 基线模型及运行
    目前支持4类代码：直接fine-tuning、PET、Ptuning、GPT
    
    直接fine-tuning: 
        一键运行.基线模型与代码
        1、克隆项目 
           git clone https://github.com/CLUEbenchmark/FewCLUE.git
        2、进入到相应的目录
           分类任务  
               例如：
               cd FewCLUE/baseline/models_tf/fine_tuning/bert/
        3、运行对应任务的脚本(GPU方式): 会自动下载模型并开始运行。
           bash run_classifier_multi_dataset.sh
           计算8个任务cecmmnt tnews iflytek ocnli csl cluewsc bustm csldcp，每个任务6个训练集的训练模型结果
           结果包括验证集和测试集的准确率，以及无标签测试集的生成提交文件


​      
​    PET/Ptuning/GPT:
​        环境准备：
​          预先安装Python 3.x(或2.7), Tesorflow 1.14+, Keras 2.3.1, bert4keras。
​          需要预先下载预训练模型：chinese_roberta_wwm_ext，并放入到pretrained_models目录下
​        
​        运行：
​        1、进入到相应的目录，运行相应的代码。以ptuning为例：
​           cd ./baselines/models_keras/ptuning
​        2、运行代码
​           python3 ptuning_iflytek.py

Zero-shot roberta版
```
环境准备：
    预先安装Python 3.x(或2.7), Tesorflow 1.14+, Keras 2.3.1, bert4keras。
    需要预先下载预训练模型：chinese_roberta_wwm_ext，并放入到pretrained_models目录下

运行：
1、在FewClue根目录运行脚本：
bash ./baselines/models_keras/zero_shot/roberta_zeroshot.sh [iflytek\tnews\eprstmt\ocnli...]
```

<a href='https://github.com/CLUEbenchmark/FewCLUE/blob/main/baselines/models_keras/gpt/readme.md'>Zero-shot gpt版</a>

1. 模型下载：    
    下载chinese_roberta_wwm_ext模型（运行gpt模型时，需要其中的vocab.txt文件，可只下载该文件）和
   <a href='https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow'> Chinese GPT模型</a>到pretrained_models目录下。

1. 运行方式：
    ```
    cd baselines/models_keras/gpt
    # -z 表示零样本学习，-t 表示不同任务名称，可替换为eprstmt,bustm,ocnli,csldcp,tnews,wsc,ifytek,csl
    python run_gpt.py -t chid -z # 运行chid任务，并使用零样本学习的方式
    ```
## KgCLUE测评


## 数据集文件结构 Data set Structure
    5份训练集，对应5份验证集，1份公开测试集，1份用于提交测试集，1份无标签样本，1份合并后的训练和验证集
    
    单个数据集目录结构：
        train_0.json：训练集0
        train_1.json：训练集1
        train_2.json：训练集2
        train_3.json：训练集3
        train_4.json：训练集4
        train_few_all.json： 合并后的训练集，即训练集0-4合并去重后的结果
        
        dev_0.json：验证集0，与训练集0对应
        dev_0.json：验证集1，与训练集1对应
        dev_0.json：验证集2，与训练集2对应
        dev_0.json：验证集3，与训练集3对应
        dev_0.json：验证集4，与训练集4对应
        dev_few_all.json： 合并后的验证集，即验证集0-4合并去重后的结果
        
        test_public.json：公开测试集，用于测试，带标签
        test.json: 测试集，用于提交，不能带标签
        
        unlabeled.json: 无标签的大量样本


## 问题 Question
    1. 问：测试系统，什么时候开发？
       答：测评系统在11月15日后才会开放。

## 贡献与参与
    1.问：我有符合代码规范的模型代码，并经过测试，可以贡献到这个项目吗？
     答：可以的。你可以提交一个pull request，并写上说明。
    
    2.问：我正在研究小样本学习，具有较强的模型研究能力，怎么参与到此项目？
      答：发送邮件到 CLUEbenchmark@163.com，标题为：参与KgCLUE课题，并介绍一下你的研究。

   添加微信入FewCLUE群:
   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/ljy.jpeg"  width="45%" height="45%" />   

   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/bq_01.jpeg"  width="45%" height="45%" />   

   QQ群:836811304

## 引用 Reference

1、<a href='https://arxiv.org/abs/2005.14165'>GPT3: Language Models are Few-Shot Learners</a>

2、<a href='https://arxiv.org/pdf/2009.07118.pdf'>PET: It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners</a>

3、<a href='https://kexue.fm/archives/7764'>必须要GPT3吗？不，BERT的MLM模型也能小样本学习</a>

4、<a href="https://arxiv.org/pdf/2012.15723.pdf">LM_BFF: Making Pre-trained Language Models Better Few-shot Learners</a>

5、<a href='https://zhuanlan.zhihu.com/p/341609647'>GPT-3的最强落地方式？陈丹琦提出小样本微调框架LM-BFF，比普通微调提升11%</a>

6、<a href='https://arxiv.org/pdf/2103.10385.pdf'>论文：GPT Understands, Too</a>

7、<a href='https://kexue.fm/archives/8295'>文章：P-tuning：自动构建模版，释放语言模型潜能</a>

8、<a href='https://arxiv.org/abs/2103.11955'>ADAPET: Improving and Simplifying Pattern Exploiting Training</a>

9、<a href='https://arxiv.org/abs/2104.14690'>EFL:Entailment as Few-Shot Learner</a>

## License

    正在添加中
## 引用
    {FewCLUE,
      title={FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark},
      author={Liang Xu, Xiaojing Lu, Chenyang Yuan, Xuanwei Zhang, Huilin Xu, Hu Yuan, Guoao Wei, Xiang Pan, Xin Tian, Libo Qin, Hu Hai},
      year={2021},
      howpublished={\url{https://arxiv.org/abs/2107.07498}},
    }
