基线模型-代码及说明

### 1.<a href='./ner_re/'>NER_RE--两阶段方案</a>

    1）进行NER识别-->2）得到候选关系-->3）相似度模型计算最佳关系-->4）利用三元组定位到答案；
    包含数据预处理、完整流程、实验效果和已经训练好的模型。

### 2. Seq2Seq+前缀树
   <a href='https://kexue.fm/archives/8802'>Seq2Seq+前缀树：检索任务新范式（以KgCLUE为例）》</a>
    
    本文介绍了检索模型的一种新方案——“Seq2Seq+前缀树”，并以KgCLUE为例给出了一个具体的baseline。
    “Seq2Seq+前缀树”的方案有着训练简单、空间占用少等优点，也有一些不足之处，总的来说算得上是一种简明的有竞争力的方案。

### 3. <a href="https://github.com/CLUEbenchmark/KgCLUEbench">KgCLUEbench</a>
我们还提供了另一个代码库来复现我们的效果
  
### 4. <a href='./other_implement/'>原始方案</a>

### 效果评估脚本
     Score=EM_O * 0.50 + F1_O * 0.50
<a href='./baselines/evaluate_f1_em.py'>evaluate_f1_em.py</a>