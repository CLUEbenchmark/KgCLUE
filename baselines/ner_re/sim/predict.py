'''
-------------------------------------------------
   Description :    sentence transformer
使用训练的相似度，针对单个样本进行预测
-------------------------------------------------

'''

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model

class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.init_model()

    def init_model(self):
        """ init classification model
            input: None
            output: None
        """
        self.tokenizer = Tokenizer(self.args.dict_path, do_lower_case=True)
        self.model = build_transformer_model(
                    config_path=self.args.config_path,
                    checkpoint_path=self.args.checkpoint_path,
                    with_pool=True,
                    return_keras_model=False,
                )
        output = Dropout(rate=0.1)(self.model.model.output)
        output = Dense(
            units=2, 
            activation='softmax', 
            kernel_initializer=self.model.initializer
        )(output)
        self.model = Model(self.model.model.input, output)
        model_path = os.path.join(self.args.output_dir, 'best_model.weights')
        self.model.load_weights(model_path)

    def convert_text_format(self, text_1, text_2):
        """ convert predict input into prompt format
            input: text
            output: token_ids, segment_ids
        """
        token_ids, segment_ids = self.tokenizer.encode(
                text_1, text_2, maxlen=self.args.maxlen
            )
        batch_token_ids, batch_segment_ids = [], []
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        return batch_token_ids, batch_segment_ids

    def predict(self, text_1, text_2):
        """ get model predict result
            input: text1, text2
            output: result with
        """
        token_ids, segment_ids = self.convert_text_format(text_1, text_2)
        possibility_sim = self.model.predict([token_ids, segment_ids])[0][1]
        # print("possibility_sim:",possibility_sim)
        # predict_result=predict.argmax(axis=1)
        # if predict_result[0] == 1:
        #     predict_result = "相似"
        # elif predict_result[0] == 0:
        #     predict_result = "不相似"
        # result = {"text_1": text_1, "text_2": text_2, "label": predict_result}
        # print("predict.predict_result:",predict_result,";result:",result)
        return possibility_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    base_model_path = 'prev_trained_model/chinese_rbt3_L-3_H-768_A-12/'
    # parser.add_argument('--gpu_id', type=str, default = "7") # -1
    parser.add_argument('--num_classes', type=str, default = 2)
    parser.add_argument('--maxlen', type=str, default = 64)
    parser.add_argument('--output_dir', type=str, default="outputs/kg_sim_output")
    parser.add_argument('--config_path', type=str, default=base_model_path+"bert_config_rbt3.json")
    parser.add_argument('--checkpoint_path', type=str, default=base_model_path+"bert_model.ckpt")
    parser.add_argument('--dict_path', type=str, default=base_model_path+"vocab.txt")
    args = parser.parse_args()

    predictor = Predictor(args)
    text_1 = "你知道鲤鱼山的名字是怎么来的吗？"
    text_2 = "得名"
    result = predictor.predict(text_1, text_2)
    print("SIM相似度模型.text_1:",text_1,";text_2:",text_2,";possibility_sim:",result)