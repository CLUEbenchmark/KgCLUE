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
1、KBQA利用的是结构化的知识，其数据来源决定了适合回答what，when 等事实性问题。

2、KBQA的研究，根据目标问题的性质，可以分为几个方向。第一个是单跳问题 (one hop) ，第二个是多跳问题 (multi hop)。单跳问题是指可以通过知识库中某一条事实三元组来回答，而多跳问题是指需要知识库中的多条事实三元组来回答。KgCLUE第一版针对的是单跳问题构建的数据集。

3、测评的主要目标是KBQA，根据KBQA任务的特点，可以考察近年来的实体识别、关系分类以及实体链接等子任务的发展。

此外，我们提供KBQA测评完善的基础设施。
从任务设定，广泛的数据集，多个有代表性的基线模型及效果对比，一键运行脚本，到测评系统等完整的基础设施。


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


## 问题 Question
    1. 问：测试系统，什么时候开发？
       答：测评系统在11月15日后才会开放。

## 贡献与参与
    1.问：我有符合代码规范的模型代码，并经过测试，可以贡献到这个项目吗？
     答：可以的。你可以提交一个pull request，并写上说明。
    
    2.问：我正在研究KBQA学习，具有较强的模型研究能力，怎么参与到此项目？
      答：发送邮件到 CLUEbenchmark@163.com，标题为：参与KgCLUE课题，并介绍一下你的研究。


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
