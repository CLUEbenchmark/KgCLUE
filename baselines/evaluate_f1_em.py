#! -*- coding: utf-8 -*-

"""
KgCLUE评估脚本。
输入两个文件（公开测试集，公开测试集上的预测文件），输出评估指标（F1 score & EM，包括SPO上的不同部分的效果）
"""

import json, pylcs


def normalize(text):
    """简单的文本标准化
    """
    return ' '.join(text.lower().split())


def f1_sim(text_a, text_b):
    """F1相似度
    说明：算出两个文本的最长公共子序列长度，然后乘2并处以两者
    长度之和。推荐用pylcs算，速度较快。
    """
    if not text_a and not text_b:
        return 0.
    else:
        lcs = pylcs.lcs(text_a, text_b)
        return 2. * lcs / (len(text_a) + len(text_b))


def evaluate(source_file, target_file, label='answer'):
    """计算EM和F1
    说明：字符串的F1是一种模糊匹配的相似度。
    """
    source_lines = open(source_file).readlines()
    target_lines = open(target_file).readlines()
    metrics = {
        '%s_%s' % (i, j): 0.
        for i in ['EM', 'F1'] for j in ['All', 'S', 'P', 'O']
    }

    for source, target in zip(source_lines, target_lines):
        # 处理预测结果
        source = json.loads(source)
        S_source, P_source, O_source = source['answer'].split(' ||| ')
        S_source = normalize(S_source)
        P_source = normalize(P_source)
        O_source = normalize(O_source)
        # 处理标准答案
        target = json.loads(target)
        S_target, P_target, O_target = target['answer'].split(' ||| ')
        S_target = normalize(S_target)
        P_target = normalize(P_target)
        O_target = normalize(O_target)
        # 对比计算
        if S_source == S_target:
            metrics['EM_S'] += 1
        if P_source == P_target:
            metrics['EM_P'] += 1
        if O_source == O_target:
            metrics['EM_O'] += 1
        if (S_source, P_source, O_source) == (S_target, P_target, O_target):
            metrics['EM_All'] += 1
        metrics['F1_S'] += f1_sim(S_source, S_target)
        metrics['F1_P'] += f1_sim(P_source, P_target)
        metrics['F1_O'] += f1_sim(O_source, O_target)
        metrics['F1_All'] += f1_sim(S_source + P_source + O_source, S_target + P_target + O_target)

    # metrics = {: v / len(target_lines) for k, v in metrics.items()} # must be real number, not str
    metrics = {k: v / len(target_lines) for k, v in metrics.items()}
    EMF1=metrics['EM_O']*0.50+metrics['F1_O']*0.50
    metrics['EMF1']=EMF1
    metrics={k:'%.3f' % (v * 100) for k,v in metrics.items()}
    return metrics

# source_file='test_public_predict.json' # 模型预测文件
# target_file='test_public.json' # 使用公开测试集进行自测
# metrics=evaluate(source_file, target_file)
# print("metrics:",metrics)