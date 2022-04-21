from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup
from custom_dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
from utils.utils_num_question import *

import torch
import torch.nn as nn
import os, sys
import numpy as np
import argparse
import copy
import math
import random


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
        special_tokens = {
            'bos_token': self.args.bos_token,
            'additional_special_tokens': [self.args.sp1_token, self.args.sp2_token]
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        
        # Load model    
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
            
        ppd = PadCollate(eos_id=self.args.eos_id)
        valid_set = CustomDataset(self.args.valid_prefix, self.args)
           
        self.valid_loader = DataLoader(valid_set, 
                                           collate_fn=ppd.pad_collate,
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            
 
            # Calculate total training steps

        
        if self.args.ckpt_name is not None:
            ckpt_path = f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt"
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
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, labels, ask_questions = batch
                input_ids, token_type_ids, labels, ask_questions =input_ids.to(self.args.device),token_type_ids.to(self.args.device),labels.to(self.args.device),ask_questions.to(self.args.device)
                input_len = len(input_ids)

                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)               
                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                shift_labels = labels[..., 1:].contiguous().view(-1)
                shift_label = []
                for i in shift_labels:
                    if i > 0:
                        shift_label.append(i)
                label = self.tokenizer.decode(shift_label, skip_special_tokens=True)
                if check_question(nltk.word_tokenize(res)):
                    print(res)
                    num_ques_model += 1
                    
                if check_question(nltk.word_tokenize(label)):
                    num_ques_valset += 1
                               
        print('Number of questiones in validate set: ' + str(num_ques_valset))
        print('Number of questiones from model: ' + str(num_ques_model))

        return num_ques_model,num_ques_valset
    
    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos-1]  # (1, V)
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

