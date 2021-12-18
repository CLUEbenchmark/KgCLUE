import argparse
import logging
import codecs
import os
import random
import numpy as np
import torch
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification,BertTokenizer,BertConfig
from transformers.data.processors.utils import DataProcessor, InputExample
from BERT_CRF import BertCrf
from transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


CRF_LABELS = ["O", "B-LOC", "I-LOC"]


def statistical_real_sentences(input_ids:torch.Tensor,mask:torch.Tensor,predict:list)-> list:
    # shape (batch_size,max_len)
    assert input_ids.shape == mask.shape
    # batch_size
    assert input_ids.shape[0] == len(predict)

    # 第0位是[CLS] 最后一位是<pad> 或者 [SEP]
    new_ids = input_ids[:,1:-1]
    new_mask = mask[:,2:]

    real_ids = []
    for i in range(new_ids.shape[0]):
        seq_len = new_mask[i].sum()
        assert seq_len == len(predict[i])
        real_ids.append(new_ids[i][:seq_len].tolist())
    return real_ids


def flatten(inputs:list) -> list:
    result = []
    [result.extend(line) for line in inputs]
    return result












def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



class CrfInputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class CrfInputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label



def crf_convert_examples_to_features(examples,tokenizer,
                                     max_length=512,
                                     label_list=None,
                                     pad_token=0,
                                     pad_token_segment_id = 0,
                                     mask_padding_with_zero = True):

    label_map = {label:i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text,
            add_special_tokens=True,
            max_length=max_length,
            truncate_first_sequence=True  # We're truncating the first sequence in priority if True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # 第一个和第二个[0] 加的是[CLS]和[SEP]的位置,  [0]*padding_length是[pad] ，把这些都暂时算作"O"，后面用mask 来消除这些，不会影响
        labels_ids = [0] + [label_map[l] for l in example.label] + [0] + [0]*padding_length



        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),max_length)

        assert len(labels_ids) == max_length, "Error with input length {} vs {}".format(len(labels_ids),max_length)


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in labels_ids]))

        features.append(
            CrfInputFeatures(input_ids,attention_mask,token_type_ids,labels_ids)
        )
    return features


class NerProcessor(DataProcessor):
    def get_train_examples(self,data_dir):
        return self._create_examples(
            os.path.join(data_dir,"train.txt"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "test.txt"))


    def get_labels(self):
        return CRF_LABELS

    @classmethod
    def _create_examples(cls, path):
        lines = []
        max_len = 0
        with codecs.open(path, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            for line in f:
                tokens = line.strip().split(' ')
                if 2 == len(tokens):
                    word = tokens[0]
                    label = tokens[1]
                    word_list.append(word)
                    label_list.append(label)
                elif 1 == len(tokens) and '' == tokens[0]:
                    if len(label_list) > max_len:
                        max_len = len(label_list)

                    lines.append((word_list,label_list))
                    word_list = []
                    label_list = []
        examples = []
        for i,(sentence,label) in enumerate(lines):
            examples.append(
                CrfInputExample(guid=i,text=" ".join(sentence),label=label)
            )
        return examples


def load_and_cache_example(args,tokenizer,processor,data_type):


    type_list = ['train', 'dev', 'test']
    if data_type not in type_list:
        raise ValueError("data_type must be one of {}".format(" ".join(type_list)))

    cached_features_file = "cached_{}_{}".format(data_type, str(args.max_seq_length))
    cached_features_file = os.path.join(args.data_dir, cached_features_file)
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if type_list[0] == data_type:
            examples = processor.get_train_examples(args.data_dir)
        elif type_list[1] == data_type:
            examples = processor.get_dev_examples(args.data_dir)
        elif type_list[2] == data_type:
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise ValueError("UNKNOW ERROR")
        features = crf_convert_examples_to_features(examples=examples,tokenizer=tokenizer,max_length=args.max_seq_length,label_list=label_list)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    return dataset


def trains(args,train_dataset,eval_dataset,model):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight','transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    best_f1 = 0.
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step,batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'tags':batch[3],
                      'decode':True
            }
            outputs = model(**inputs)
            loss,pre_tag = outputs[0], outputs[1]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
            logging_loss += loss.item()
            tr_loss += loss.item()
            if 0 == (step + 1) % args.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                logger.info("EPOCH = [%d/%d] global_step = %d   loss = %f",_+1,args.num_train_epochs,global_step,
                            logging_loss)
                logging_loss = 0.0

                # if (global_step < 100 and global_step % 10 == 0) or (global_step % 50 == 0):
                # 每 相隔 100步，评估一次
                if global_step % 100 == 0:
                    best_f1 = evaluate_and_save_model(args,model,eval_dataset,_,global_step,best_f1)

    # 最后循环结束 再评估一次
    best_f1 = evaluate_and_save_model(args, model, eval_dataset,_,global_step, best_f1)




def evaluate_and_save_model(args,model,eval_dataset,epoch,global_step,best_f1):
    ret = evaluate(args, model, eval_dataset)

    precision_b = ret['1']['precision']
    recall_b = ret['1']['recall']
    f1_b = ret['1']['f1-score']
    support_b = ret['1']['support']

    precision_i = ret['2']['precision']
    recall_i = ret['2']['recall']
    f1_i = ret['2']['f1-score']
    support_i = ret['2']['support']

    weight_b = support_b / (support_b + support_i)
    weight_i = 1 - weight_b

    avg_precision = precision_b * weight_b + precision_i * weight_i
    avg_recall = recall_b * weight_b + recall_i * weight_i
    avg_f1 = f1_b * weight_b + f1_i * weight_i

    all_avg_precision = ret['micro avg']['precision']
    all_avg_recall = ret['micro avg']['recall']
    all_avg_f1 = ret['micro avg']['f1-score']

    logger.info("Evaluating EPOCH = [%d/%d] global_step = %d", epoch+1,args.num_train_epochs,global_step)
    logger.info("B-LOC precision = %f recall = %f  f1 = %f support = %d", precision_b, recall_b, f1_b,
                support_b)
    logger.info("I-LOC precision = %f recall = %f  f1 = %f support = %d", precision_i, recall_i, f1_i,
                support_i)

    logger.info("attention AVG:precision = %f recall = %f  f1 = %f ", avg_precision, avg_recall,
                avg_f1)
    logger.info("all AVG:precision = %f recall = %f  f1 = %f ", all_avg_precision, all_avg_recall,
                all_avg_f1)

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_ner.bin"))
        logging.info("save the best model %s,avg_f1= %f", os.path.join(args.output_dir, "best_bert.bin"),
                     best_f1)
    # 返回出去，用于更新外面的 最佳值
    return best_f1






def evaluate(args, model, eval_dataset):

    eval_output_dirs = args.output_dir
    if not os.path.exists(eval_output_dirs):
        os.makedirs(eval_output_dirs)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    loss = []
    real_token_label = []
    pred_token_label = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'tags':batch[3],
                      'decode':True,
                      'reduction':'none'
            }
            outputs = model(**inputs)
            # temp_eval_loss shape: (batch_size)
            # temp_pred : list[list[int]] 长度不齐
            temp_eval_loss, temp_pred = outputs[0], outputs[1]

            loss.extend(temp_eval_loss.tolist())
            pred_token_label.extend(temp_pred)
            real_token_label.extend(statistical_real_sentences(batch[3],batch[1],temp_pred))


    loss = np.array(loss).mean()
    real_token_label = np.array(flatten(real_token_label))
    pred_token_label = np.array(flatten(pred_token_label))
    assert real_token_label.shape == pred_token_label.shape
    ret = classification_report(y_true = real_token_label,y_pred = pred_token_label,output_dict = True)
    model.train()
    return ret






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="数据文件目录，因当有train.txt dev.txt")

    parser.add_argument("--vob_file", default=None, type=str, required=True,
                        help="词表文件")

    parser.add_argument("--model_config", default=None, type=str, required=True,
                        help="模型配置文件json文件")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="输出结果的文件")

    # Other parameters
    parser.add_argument("--pre_train_model", default=None, type=str, required=False,
                        help="预训练的模型文件，参数矩阵。如果存在就加载")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="输入到bert的最大长度，通常不应该超过512")
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="训练集的batch_size")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="验证集的batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="最大的梯度更新")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="epoch 数目")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="让学习增加到1的步数，在warmup_steps后，再衰减到0")

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    #          filename='./output/bert-crf-ner.log',
    processor = NerProcessor()

    # 得到tokenizer
    tokenizer_inputs = ()
    tokenizer_kwards = {'do_lower_case': False,
                        'max_len': args.max_seq_length,
                        'vocab_file': args.vob_file}
    tokenizer = BertTokenizer(*tokenizer_inputs,**tokenizer_kwards)


    model = BertCrf(config_name= args.model_config,model_name=args.pre_train_model,num_tags = len(processor.get_labels()),batch_first=True)
    model = model.to(args.device)


    train_dataset = load_and_cache_example(args,tokenizer,processor,'train')
    eval_dataset = load_and_cache_example(args,tokenizer,processor,'dev')
    test_dataset = load_and_cache_example(args, tokenizer, processor, 'test')

    if args.do_train:
        trains(args,train_dataset,eval_dataset,model)



if __name__ == '__main__':
    main()
