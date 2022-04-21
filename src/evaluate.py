from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup
from custom_dataset_new import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
import gensim

import torch
import torch.nn as nn
import os, sys
import numpy as np
import argparse
import copy
import math
import random
from utils.utils_bleau import *
from utils.utils_distinc_n import *
from utils.utils_f1 import *
from utils.utils_num_question import *

def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    out_dims = (num, 512*786)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 0
    return out_tensor


class MainModel(nn.Module):
    def __init__(self, args, topic_feature_dim):
        super().__init__()
        self.args = args
        self.gpt = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)    
        self.gpt.resize_token_embeddings(self.args.vocab_size)
        self.fc = nn.Linear(512*786, topic_feature_dim).to(self.args.device)

    def forward(self, input_ids, token_type_ids, labels):
        gpt_outputs = self.gpt(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True, return_dict=True)
        _, logits, hidden_state = gpt_outputs[0], gpt_outputs[1], gpt_outputs[3][0].reshape(self.args.batch_size, -1)
        hidden_state = padding_tensor(hidden_state)
        topic_embedding = self.fc(hidden_state).to(self.args.device)
        return topic_embedding, gpt_outputs

class Manager():
    def __init__(self, args):
        self.args = args
        
        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        # if self.args.w_topic_loss:
        #     print("Loading the Topic Modelling model...")
        #     self.topic_model = gensim.models.LdaModel.load(self.args.topic_model_ckpt_path)
        #     self.id2word = self.topic_model.id2word
        special_tokens = {
            'bos_token': self.args.bos_token,
            'pad_token': self.args.pad_token,
            'additional_special_tokens': [self.args.sp1_token, self.args.sp2_token]
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.pad_id = vocab[self.args.pad_token]
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        # Load model    
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        # self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        # self.model.resize_token_embeddings(self.args.vocab_size)
        self.model = MainModel(self.args, topic_feature_dim=30)
        self.args.max_len = min(self.args.max_len, self.model.gpt.config.n_ctx)

        # self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
            
        valid_set = CustomDataset(self.args.valid_prefix, self.args)
        ppd = PadCollate(eos_id=self.args.eos_id, pad_id=self.args.pad_id)
            

        self.valid_loader = DataLoader(valid_set, 
                                           collate_fn=ppd.pad_collate,
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
           
        if self.args.ckpt_name is not None:
            ckpt_path = f"{self.args.ckpt_name}"
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.args.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                
                print('The evaluation will start')
            else:
                print(f"Cannot fine the specified checkpoint {ckpt_path}")
                exit()

        print("Setting finished.")
   
    
    def validation(self):
        print("Validation number of questiones processing...")
        self.model.eval()
        num_ques_model = 0
        num_ques_valset = 0
        f1 = 0
        bleau_score_1 = 0
        bleau_score_2 = 0
        distinct_1 = 0
        distinct_2 = 0
        self.fix_seed(self.args.seed)
        with torch.no_grad():
            input_hists = []

            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, labels, ask_questions = batch
                input_ids, token_type_ids, labels, ask_questions =input_ids.to(self.args.device),token_type_ids.to(self.args.device),labels.to(self.args.device),ask_questions.to(self.args.device)
                input_len = len(input_ids)
                
                input_sentence = self.tokenizer.decode(input_ids.view(-1), skip_special_tokens=True)
                input_ids = [self.args.sp1_id] + self.tokenizer.encode(input_sentence)
                input_hists.append(input_ids)
                
                if len(input_hists) >= self.args.max_turns:
                    num_exceeded = len(input_hists) - self.args.max_turns + 1
                    input_hists = input_hists[num_exceeded:]
                    
                input_ids = [self.args.bos_id] + list(chain.from_iterable(input_hists)) + [self.args.sp2_id]
                start_sp_id = input_hists[0][0]
                next_sp_id = self.args.sp1_id if start_sp_id == self.args.sp2_id else self.args.sp2_id
                assert start_sp_id != next_sp_id
                token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
                assert len(token_type_ids) == len(input_hists)
                token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.args.sp2_id]
                assert len(input_ids) == len(token_type_ids)
                input_len = len(input_ids)
                
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args.device)
                token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args.device)
                
                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len, labels)                
                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                shift_labels = labels[..., 1:].contiguous().view(-1)
                shift_label = []
                for idx in shift_labels:
                    if idx > 0:
                        shift_label.append(idx)
                label = self.tokenizer.decode(shift_label, skip_special_tokens=True)

                if check_question(nltk.word_tokenize(res)):
                    num_ques_model += 1
                    
                if check_question(nltk.word_tokenize(label)):
                    num_ques_valset += 1

                f1 += prec_recall_f1_score(res,label)
                bleau_score_1 +=  bleau_score(res,label)
                bleau_score_2 += bleau_score(res,label,2)
                distinct_1 += distinct_n_sentence_level(res,1)
                distinct_2 += distinct_n_sentence_level(res,2)
                               
        print('Number of questiones in validate set: ' + str(num_ques_valset))
        print('Number of questiones from model: ' + str(num_ques_model))
        print('F1: {:.5f}'.format(f1/(i+1)))
        print('Bleau Score 1: {:.5f}'.format(bleau_score_1/(i+1)))
        print('Bleau Score 2: {:.5f}'.format(bleau_score_2/(i+1)))
        print('Distinct 1: {:.5f}'.format(distinct_1/(i+1)))
        print('Distinct 2: {:.5f}'.format(distinct_2/(i+1)))

        return num_ques_model,num_ques_valset
    
    def nucleus_sampling(self, input_ids, token_type_ids, input_len, labels):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            output = self.model.gpt(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos-1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.args.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)
            
            probs = torch.zeros(output.shape, device=self.args.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)
            
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)
            
            if idx_item == self.args.eos_id:
                break
                
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.args.sp2_id]]).to(self.args.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape
            
        return output_ids
    
    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
                    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--data_dir', type=str, default="data", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument('--model_type', type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument('--bos_token', type=str, default="<bos>", help="The BOS token.")
    parser.add_argument('--sp1_token', type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument('--sp2_token', type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    parser.add_argument('--batch_size', type=int, default=1, help="The batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument('--max_turns', type=int, default=5, help="The maximum number of dialogue histories to include.")
    parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--ckpt_dir', type=str, default="saved_models", help="The directory name for saved checkpoints.")
    parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")           
    parser.add_argument('--w_question_loss', type=float, default=0.25, help="The weight value of question loss.")
    parser.add_argument('--w_topic_loss', type=float, default=0.25, help="The weight value of topic loss.")
    parser.add_argument('--topic_model_ckpt_path', type=str, default='saved_models/topic_modelling/mymodel', help="The path of the trained checkpoint of the topic model.")
    parser.add_argument('--pad_token', type=str, default="<pad>", help="The PAD token.")

    args = parser.parse_args()
    
    assert args.model_type in [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"
    ]
    
    args.data_dir = f"{args.data_dir}/{args.model_type}"
    args.ckpt_dir = f"{args.ckpt_dir}/{args.model_type}"
              
    assert args.ckpt_name is not None, "Please specify the trained model checkpoint."

    manager = Manager(args)
    manager.validation()