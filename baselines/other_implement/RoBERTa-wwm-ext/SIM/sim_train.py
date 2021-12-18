#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 11:13
# @Author  : 刘鑫
# @FileName: sequnce_labeling_train.py
# @Software: PyCharm

# 假定已经带着正确的分类label去训练实体识别
import tensorflow as tf
import os, math, json

from frame.bert import modeling
from frame.bert import optimization

from config import config as cf

config = {
    "in_1": r"C:\Users\11943\Documents\GitHub\RoBERTa-wwm-ext\SIM\data\train\train.tf_record",  # 第一个输入为 训练文件
    "in_2": r"C:\Users\11943\Documents\GitHub\RoBERTa-wwm-ext\SIM\data\eval\eval.tf_record",  # 第二个输入为 验证文件
    "bert_config": cf.bert_config,  # bert模型配置文件
    "init_checkpoint": cf.init_checkpoint,  # 预训练bert模型
    "vocab_file": cf.vocab_file,
    "train_examples_len": cf.sim_train_nums,
    "dev_examples_len": cf.sim_valid_nums,
    "num_labels": 2,
    "train_batch_size": 8,
    "dev_batch_size": 8,
    "num_train_epochs": 5,
    "eval_start_step": 1300,
    "eval_per_step": 300,
    "auto_save": 300,
    "learning_rate": 3e-5,
    "warmup_proportion": 0.1,
    "out": "./SIM/model/",  # 保存模型路径
    "out_1": "./SIM/model/",  # 保存模型路径
    "train_out": "./SIM/train_out/"
}


def load_bert_config(path):
    """
    bert 模型配置文件
    """
    return modeling.BertConfig.from_json_file(path)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, keep_prob, num_labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope='bert'
    )
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits_wx = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits_wx, output_bias)

        probabilities = tf.sigmoid(logits)
        label_ids = tf.cast(labels, tf.float32)
        per_example_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_ids), axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        predict_ids = tf.cast(probabilities > 0.5, tf.int32)
        label_ids = tf.cast(label_ids, tf.int32)
        elements_equal = tf.cast(tf.equal(predict_ids, label_ids), tf.int32)
        row_predict_ids = tf.reduce_sum(elements_equal, -1)
        row_label_ids = tf.reduce_sum(tf.ones_like(label_ids), -1)
        _, accuracy = tf.metrics.accuracy(labels=row_label_ids, predictions=row_predict_ids)

        return (loss, accuracy, logits, probabilities)


def get_input_data(input_file, seq_length, batch_size, is_training=True):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([2], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_ids"]
        input_mask = example["input_mask"]
        segment_ids = example["segment_ids"]
        labels = example["label_ids"]
        return input_ids, input_mask, segment_ids, labels

    dataset = tf.data.TFRecordDataset(input_file)
    # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
    if is_training:
        dataset = dataset.map(parser).batch(batch_size).shuffle(buffer_size=3000)
    else:
        dataset = dataset.map(parser).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, labels = iterator.get_next()
    return input_ids, input_mask, segment_ids, labels


def main():
    print("print start load the params...")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(config["out"])
    # 配置超参数
    train_examples_len = config["train_examples_len"]
    dev_examples_len = config["dev_examples_len"]
    learning_rate = config["learning_rate"]
    eval_per_step = config["eval_per_step"]
    num_labels = config["num_labels"]
    num_train_steps = math.ceil(train_examples_len / config["train_batch_size"])
    num_dev_steps = math.ceil(dev_examples_len / config["dev_batch_size"])
    num_warmup_steps = math.ceil(num_train_steps * config["num_train_epochs"] * config["warmup_proportion"])
    print("num_train_steps:{},  num_dev_steps:{},  num_warmup_steps:{}".format(num_train_steps, num_dev_steps,
                                                                               num_warmup_steps))

    use_one_hot_embeddings = False
    is_training = True
    use_tpu = False
    seq_len = cf.sim_max_seq_length
    init_checkpoint = config["init_checkpoint"]
    print("print start compile the bert model...")

    # 定义输入输出
    input_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids')
    input_mask = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask')
    segment_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids')
    token_labels = tf.placeholder(tf.int64, shape=[None, 2], name='token_labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'
    bert_config_ = load_bert_config(config["bert_config"])
    (total_loss, acc, logits, probabilities) = create_model(bert_config_, is_training, input_ids,
                                                            input_mask, segment_ids, token_labels,
                                                            keep_prob,
                                                            num_labels, use_one_hot_embeddings)
    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps * config["num_train_epochs"],
                                             num_warmup_steps, False)
    print("print start train the bert model...")

    batch_size = config["train_batch_size"]
    dev_batch_size = config["dev_batch_size"]
    init_global = tf.global_variables_initializer()
    # print([v for v in tf.global_variables()])
    saver = tf.train.Saver([v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name],
                           max_to_keep=3)  # 保存最后top3模型

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init_global)
        print("start load the pre train model")

        if init_checkpoint:
            tvars = tf.trainable_variables()
            print("trainable_variables", len(tvars))
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)

            print("initialized_variable_names:", len(initialized_variable_names))
            saver_ = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
            saver_.restore(sess, init_checkpoint)

            tvars = tf.global_variables()
            print("global_variables", len(tvars))
            initialized_vars = [v for v in tvars if v.name in initialized_variable_names]
            not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
            tf.logging.info('--all size %s; initialized size %s' % (len(tvars), len(initialized_vars)))
            tf.logging.info('--all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            for v in initialized_vars:
                print('--initialized: %s, shape = %s' % (v.name, v.shape))
            for v in not_initialized_vars:
                print('--not initialized: %s, shape = %s' % (v.name, v.shape))

        else:
            sess.run(tf.global_variables_initializer())
        print("********* train start *********")

        def train_step(ids, mask, segment, y, step, train_out_f):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    token_labels: y,
                    keep_prob: 0.9}
            _, out_loss, acc_, p_ = sess.run([train_op, total_loss, acc, probabilities], feed_dict=feed)
            print("step :{},loss :{}, acc :{}".format(step, out_loss, acc_))
            train_out_f.write("step :{}, loss :{},  acc :{} \n".format(step, out_loss, acc_))
            return out_loss, p_, y

        def dev_step(ids, mask, segment, y):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    token_labels: y,
                    keep_prob: 1.0
                    }
            out_loss, acc_, p_ = sess.run([total_loss, acc, probabilities], feed_dict=feed)
            print("loss :{}, acc :{}".format(out_loss, acc_))
            return out_loss, p_, y

        min_total_loss_dev = 999999
        step = 0
        if not os.path.exists(config["train_out"]):
            os.makedirs(config["train_out"])
        for epoch in range(config["num_train_epochs"]):
            _ = "{:*^100s}".format(("epoch-" + str(epoch)).center(20))
            print(_)
            # 读取训练数据
            total_loss_train = 0

            input_ids2, input_mask2, segment_ids2, labels2 = get_input_data(config["in_1"], seq_len, batch_size)

            if not os.path.exists(os.path.join(config["train_out"])):
                os.makedirs(os.path.join(config["train_out"]))

            train_out_f = open(os.path.join(config["train_out"], "epoch-" + str(epoch) + ".txt"), 'w', encoding='utf-8')

            for i in range(num_train_steps):
                step += 1
                ids_train, mask_train, segment_train, y_train = sess.run(
                    [input_ids2, input_mask2, segment_ids2, labels2])
                out_loss, pre, y = train_step(ids_train, mask_train, segment_train, y_train, step, train_out_f)
                total_loss_train += out_loss

                if step % eval_per_step == 0 and step >= config["eval_start_step"]:
                    total_loss_dev = 0
                    dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2 = get_input_data(config["in_2"],
                                                                                                    seq_len,
                                                                                                    dev_batch_size,
                                                                                                    False)

                    for j in range(num_dev_steps):  # 一个 epoch 的 轮数
                        ids_dev, mask_dev, segment_dev, y_dev = sess.run(
                            [dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2])
                        out_loss, pre, y = dev_step(ids_dev, mask_dev, segment_dev, y_dev)
                        total_loss_dev += out_loss

                    print("total_loss_dev:{}".format(total_loss_dev))
                    # print(classification_report(total_true_dev, total_pre_dev, digits=4))

                    if total_loss_dev < min_total_loss_dev:
                        print("save model:\t%f\t>%f" % (min_total_loss_dev, total_loss_dev))
                        min_total_loss_dev = total_loss_dev
                        saver.save(sess, config["out"] + 'bert.ckpt', global_step=step)
                elif step < config["eval_start_step"] and step % config["auto_save"] == 0:
                    saver.save(sess, config["out"] + 'bert.ckpt', global_step=step)
            train_out_f.close()
            _ = "{:*^100s}".format(("epoch-" + str(epoch) + " report:").center(20))
            print("total_loss_train:{}".format(total_loss_train))
            # print(classification_report(total_true_train, total_pre_train, digits=4))
    sess.close()


if __name__ == "__main__":
    print("********* ner start *********")
    main()
