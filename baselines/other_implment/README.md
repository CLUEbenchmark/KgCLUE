# 初始版本（inti implemnt）的的说明


## 实现思路
下面简单介绍该任务的baseline的构建思路，但并不对任务数据进行详细介绍，如对任务数据不明白，请返回[数据集介绍](#数据集介绍)部分

以下图片仅为思路参考

bert-base-chinese版本

#### NER阶段
 我们使用BertForTokenClassification + crf来做NER任务，如图所示用于识别出问题中的实体
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForTokenClassification+crf.png"  width="100%" height="100%" /> 

#### SIM阶段 
我们使用BertForSequenceClassification，用于句子分类。把问题和属性拼接到一起，用于判断问题要问的是不是这个属性
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForSequenceClassification.png"  width="85%" height="85%" /> 

#### chinese-roberta-wwm-ext-large以及chinese-roberta-wwm-ext模型
我们同样适用NER+SIM的思路
但在NER阶段采用了BERT+LSTM+CRF用于识别问题中的实体。输入是wordPiece tokenizer得到的tokenid，进入Bert预训练模型抽取丰富的文本特征得到batch_size * max_seq_len * emb_size的输出向量，输出向量过Bi-LSTM从中提取实体识别所需的特征，得到batch_size * max_seq_len * (2*hidden_size)的向量，最终进入CRF层进行解码，计算最优的标注序列。
如图所示
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/Bert-Bilstm-CRF.png"  width="100%" height="100%" /> 

SIM阶段采用BERT作二分类模型


## 实验结果
实验设置：训练集和验证集使用训练集与验证集，测试集使用公开测试集。
###### 实验结果与线上结果不一致。线上采用新的指标进行测试，代码见<a href='../evaluate_f1_em.py'>这里</a>


| Model   | F1     | EM  |
| :----:| :----:  |:----:  |
| Bert-base-chinese |  81.8      |   79.1   |
| chinese-roberta-wwm-ext-large |  82.6     |  80.6    |
| chinese-roberta-wwm-ext |  82.3        |  80.2    |

| Model   | NER（F1-score）    | SIM(F1-score)  |
| :----:| :----:  |:----:  |
| Bert-base-chinese |  76.1      |   74.8   |
| chinese-roberta-wwm-ext-large |  81.3     |  76.93    |
| chinese-roberta-wwm-ext |  80.4     |  75.5   |



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



## 实现思路
下面简单介绍该任务的baseline的构建思路，但并不对任务数据进行详细介绍，如对任务数据不明白，请返回[数据集介绍](#数据集介绍)部分

以下图片仅为思路参考

bert-base-chinese版本

#### NER阶段
 我们使用BertForTokenClassification + crf来做NER任务，如图所示用于识别出问题中的实体
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForTokenClassification+crf.png"  width="100%" height="100%" /> 

#### SIM阶段 
我们使用BertForSequenceClassification，用于句子分类。把问题和属性拼接到一起，用于判断问题要问的是不是这个属性
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/BertForSequenceClassification.png"  width="85%" height="85%" /> 

#### chinese-roberta-wwm-ext-large以及chinese-roberta-wwm-ext模型
我们同样适用NER+SIM的思路
但在NER阶段采用了BERT+LSTM+CRF用于识别问题中的实体。输入是wordPiece tokenizer得到的tokenid，进入Bert预训练模型抽取丰富的文本特征得到batch_size * max_seq_len * emb_size的输出向量，输出向量过Bi-LSTM从中提取实体识别所需的特征，得到batch_size * max_seq_len * (2*hidden_size)的向量，最终进入CRF层进行解码，计算最优的标注序列。
如图所示
<img src="https://github.com/CLUEbenchmark/KgCLUE/blob/main/resources/img/Bert-Bilstm-CRF.png"  width="100%" height="100%" /> 

SIM阶段采用BERT作二分类模型



## 基线模型及运行 
>我们提供了另一个代码库可以更简单方便的复现我们的效果https://github.com/CLUEbenchmark/KgCLUEbench
### 数据处理
```：
1、运行baseline/dataProcessing/NERdata.py 处理NER数据
2、运行baseline/dataProcessing/SIMdata.py 处理SIM数据
``` 
####  模型  
#### 1）bert-base-chinese模型：

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

#### 2）chinese-roberta-wwm-ext-large模型：
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


#### 3）chinese-roberta-wwm-ext模型：
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

#### 4）如何测试：
1、进入到相对应的目录：
```
cd ./baseline/Evaluation
```
2、修改下列指令：
```
python run_squad.py  \
    --model_type bert   \
    --model_name_or_path BERTmodel  \
    --output_dir model \
    --data_dir data/  \
    --predict_file    \
    --do_eval   \
    --version_2_with_negative \
    --do_lower_case  \
    --per_gpu_eval_batch_size 12   \
    --max_seq_length 384   \
    --doc_stride 128
```
3、运行上述指令
