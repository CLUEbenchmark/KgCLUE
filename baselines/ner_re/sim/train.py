#! -*- coding:utf-8 -*-
# 句子对分类任务，KgCLUE数据集
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf

import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import json

set_gelu('tanh')  # 切换gelu版本

maxlen = 64
batch_size = 128
num_epoch=12 # 4
base_path='prev_trained_model/chinese_rbt3_L-3_H-768_A-12/'
config_path = base_path+'bert_config_rbt3.json'
checkpoint_path = base_path+'bert_model.ckpt'
dict_path = base_path+'vocab.txt'
output_dir='outputs/kg_sim_output/'
PROCESS_DIR='./processed_data/' # 需要的训练数据所在的目录

def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            # {"question": "楠迪·宗拉维蒙家里有什么人？", "relationship": "塔高", "label": "0"}
            # text1, text2, label = l.strip().split('\t')
            json_string=json.loads(l.strip())
            text1, text2, label= json_string["question"],json_string["relationship"],json_string["label"]
            D.append((text1, text2, int(label)))
    return D

# 加载数据集
train_data = load_data(PROCESS_DIR+'train.json')
valid_data = load_data(PROCESS_DIR+'dev.json')
test_data = load_data(PROCESS_DIR+'test.json')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_weights(output_dir+'best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=num_epoch, #
        callbacks=[evaluator]
    )

    model.load_weights(output_dir+'best_model.weights')
    print(u'Simility Mode.final test acc: %05f\n' % (evaluate(test_generator)))

else:

    model.load_weights(output_dir+'best_model.weights')