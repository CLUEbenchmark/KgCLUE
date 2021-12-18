from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from SIM import SimProcessor,SimInputFeatures,cal_acc
import torch
from tqdm import tqdm, trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SimProcessor()
tokenizer_inputs = ()
tokenizer_kwards = {'do_lower_case': False,
                    'max_len': 64,
                    'vocab_file': './input/config/bert-base-chinese-vocab.txt'}
tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)

features = torch.load('./input/data/sim_data/cached_dev_64')

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
all_label = torch.tensor([f.label for f in features], dtype=torch.long)
test_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)


bert_config = BertConfig.from_pretrained('./input/config/bert-base-chinese-config.json')
bert_config.num_labels = len(processor.get_labels())

model = BertForSequenceClassification(bert_config)
model.load_state_dict(torch.load('./output/best_sim.bin'))
model = model.to(device)


test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler,batch_size=256)

total_loss = 0.       # loss 的总和
total_sample_num = 0  # 样本总数目
all_real_label = []   # 记录所有的真实标签列表
all_pred_label = []   # 记录所有的预测标签列表

for batch in tqdm(test_dataloader, desc="testing"):
    model.eval()
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3],
                  }
        outputs = model(**inputs)
        loss, logits = outputs[0], outputs[1]

        total_loss += loss * batch[0].shape[0]  # loss * 样本个数
        total_sample_num += batch[0].shape[0]  # 记录样本个数

        pred = logits.argmax(dim=-1).tolist()  # 得到预测的label转为list

        all_pred_label.extend(pred)  # 记录预测的 label
        all_real_label.extend(batch[3].view(-1).tolist())  # 记录真实的label
loss = total_loss / total_sample_num
question_acc,label_acc = cal_acc(all_real_label,all_pred_label)

print("loss",loss.item())
print("question_acc",question_acc)
print("label_acc",label_acc)

