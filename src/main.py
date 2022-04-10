from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup
from custom_dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain

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
        if self.args.w_topic_loss:
            if self.args.topic_model == "LDA":
                import gensim
                print("Loading the LDA Topic Modelling model...")
                self.topic_model = gensim.models.LdaModel.load('saved_models/topic_modelling/lda/mymodel')
                self.id2word = self.topic_model.id2word
            elif self.args.topic_model == "BERT":
                from bertopic import BERTopic
                print("Loading the BERT Topic Modelling model...")
                self.topic_model = BERTopic.load('saved_models/topic_modelling/bert/model_w_prob.bertopic')
                self.args.train_topic_epochs = 1
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
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
            
        if self.args.mode == 'train':            
            # Load optimizer
            print("Loading the optimizer...")
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            self.best_loss = sys.float_info.max
            self.last_epoch = 0
            
            # Load train & valid dataset
            print("Loading train & valid data...")
            train_set = CustomDataset(self.args.train_prefix, self.args)
            valid_set = CustomDataset(self.args.valid_prefix, self.args)
            ppd = PadCollate(eos_id=self.args.eos_id, pad_id=self.args.pad_id)
            
            self.train_loader = DataLoader(train_set, 
                                           collate_fn=ppd.pad_collate, 
                                           shuffle=True, 
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            self.valid_loader = DataLoader(valid_set, 
                                           collate_fn=ppd.pad_collate,
                                           batch_size=self.args.batch_size, 
                                           num_workers=self.args.num_workers, 
                                           pin_memory=True)
            
            if not os.path.exists(self.args.ckpt_dir):
                os.makedirs(self.args.ckpt_dir)
                
            # Calculate total training steps
            num_batches = len(self.train_loader)
            args.total_train_steps = args.num_epochs * num_batches
            args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)
            
            self.sched = get_polynomial_decay_schedule_with_warmup(
                self.optim,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.total_train_steps,
                power=2
            )
            
            self.writer = SummaryWriter(log_dir=f'runs/{self.args.exp_name}')
        
        if self.args.ckpt_name is not None:
            ckpt_path = f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt"
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.args.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                
                if self.args.mode == 'train':
                    print(f"The training restarts with the specified checkpoint: {self.args.ckpt_name}.ckpt.")
                    self.optim.load_state_dict(ckpt['optim_state_dict'])
                    self.sched.load_state_dict(ckpt['sched_state_dict'])
                    self.best_loss = ckpt['loss']
                    self.last_epoch = ckpt['epoch']
                else:
                    print("The inference will start with the specified checkpoint.")
            else:
                print(f"Cannot fine the specified checkpoint {ckpt_path}.")
                if self.args.mode == 'train':
                    print("Training will start with the initialized model.")
                else:
                    print("Cannot inference.")
                    exit()
              
        print("Setting finished.")
              
    def train(self):
        self.fix_seed(self.args.seed)  # Fix seed before training
        print("Training starts.")
        
        start_epoch = self.last_epoch+1
        for epoch in range(start_epoch, start_epoch+self.args.num_epochs+1):
            self.model.train()
            
            print(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
            if self.args.w_topic_loss:
                if (self.args.num_epochs - epoch < self.args.train_topic_epochs):
                    self.curr_w_topic_loss = self.args.w_topic_loss
                else:
                    self.curr_w_topic_loss = 0
            else:
                self.curr_w_topic_loss = 0
            train_losses = []
            train_lm_losses = []
            train_question_losses = []
            train_topic_losses = []
            train_ppls = []
            tqdm._instances.clear()
            for i, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, labels, ask_questions = batch
                input_ids, token_type_ids, labels, ask_questions = \
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), labels.to(self.args.device), ask_questions.to(self.args.device)
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = labels
                )
                _, logits = outputs[0], outputs[1]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                criteria = nn.CrossEntropyLoss(ignore_index=self.args.pad_id)
                lm_loss = criteria(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if self.args.w_question_loss:
                    ask_question_loss = self.compute_question_loss(logits=logits, ask_question_labels=ask_questions)
                else:
                    ask_question_loss = torch.tensor(0.)
                if self.curr_w_topic_loss:
                    topic_loss = self.compute_topic_loss(logits=logits, labels=labels)
                else:
                    topic_loss = torch.tensor(0.)
                loss = lm_loss * (1 - self.args.w_question_loss - self.curr_w_topic_loss) + ask_question_loss * self.args.w_question_loss + topic_loss * self.curr_w_topic_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.sched.step()
                
                train_losses.append(loss.detach())
                train_lm_losses.append(lm_loss.detach())
                train_question_losses.append(ask_question_loss.detach())
                train_topic_losses.append(topic_loss.detach())
                ppl = torch.exp(lm_loss.detach())
                train_ppls.append(ppl)
            
            train_losses = [loss.item() for loss in train_losses]
            train_lm_losses = [loss.item() for loss in train_lm_losses]
            train_question_losses = [loss.item() for loss in train_question_losses]
            train_topic_losses = [loss.item() for loss in train_topic_losses]
            train_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls]
            train_loss = np.mean(train_losses)
            train_lm_loss = np.mean(train_lm_losses)
            train_question_loss = np.mean(train_question_losses)
            train_topic_loss = np.mean(train_topic_losses)
            train_ppl = np.mean(train_ppls)
            print(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")
            print(f"Train LM loss: {train_lm_loss} || Train Question loss: {train_question_loss} || Train Topic loss: {train_topic_loss}")
            
            self.writer.add_scalar("WeightedSumLoss/train", train_loss, epoch)
            self.writer.add_scalar("LMLoss/train", train_lm_loss, epoch)
            self.writer.add_scalar("QuestionLoss/train", train_question_loss, epoch)
            self.writer.add_scalar("TopicLoss/train", train_topic_loss, epoch)
            self.writer.add_scalar("PPL/train", train_ppl, epoch)
            
            self.last_epoch += 1
            
            valid_loss, valid_ppl, valid_lm_loss, valid_question_loss, valid_topic_loss = self.validation()
              
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'sched_state_dict': self.sched.state_dict(),
                    'loss': self.best_loss,
                    'epoch': self.last_epoch
                }
              
                torch.save(state_dict, f"{self.args.ckpt_dir}/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
                print("*"*10 + "Current best checkpoint is saved." + "*"*10)
                print(f"{self.args.ckpt_dir}/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
              
            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")
            print(f"Valid LM loss: {valid_lm_loss} || Valid Question loss: {valid_question_loss} || Valid Topic loss: {valid_topic_loss}")
            
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("LMLoss/valid", valid_lm_loss, epoch)
            self.writer.add_scalar("QuestionLoss/valid", valid_question_loss, epoch)
            self.writer.add_scalar("TopicLoss/valid", valid_topic_loss, epoch)
            self.writer.add_scalar("PPL/valid", valid_ppl, epoch)
            
            self.writer.add_scalars("Losses", {
                'train': train_loss, 
                'valid': valid_loss,
            }, epoch)
            self.writer.add_scalars("LMLosses", {
                'train': train_lm_loss, 
                'valid': valid_lm_loss,
            }, epoch)
            self.writer.add_scalars("QuestionLosses", {
                'train': train_question_loss, 
                'valid': valid_question_loss,
            }, epoch)
            self.writer.add_scalars("TopicLosses", {
                'train': train_topic_loss, 
                'valid': valid_topic_loss,
            }, epoch)
            self.writer.add_scalars("PPLs", {
                'train': train_ppl,
                'valid': valid_ppl,
            }, epoch)
              
        print("Training finished!")
    
    def validation(self):
        print("Validation processing...")
        self.model.eval()
              
        valid_losses = []
        valid_lm_losses = []
        valid_question_losses = []
        valid_topic_losses = []
        valid_ppls = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, labels, ask_questions = batch
                input_ids, token_type_ids, labels, ask_questions = \
                    input_ids.to(self.args.device), token_type_ids.to(self.args.device), labels.to(self.args.device), ask_questions.to(self.args.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    labels = labels
                )
                
                _, logits = outputs[0], outputs[1]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                criteria = nn.CrossEntropyLoss(ignore_index=self.args.pad_id)
                lm_loss = criteria(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if self.args.w_question_loss:
                    ask_question_loss = self.compute_question_loss(logits=logits, ask_question_labels=ask_questions)
                else:
                    ask_question_loss = torch.tensor(0.)
                if self.curr_w_topic_loss:
                    topic_loss = self.compute_topic_loss(logits=logits, labels=labels)
                else:
                    topic_loss = torch.tensor(0.)
                loss = lm_loss * (1 - self.args.w_question_loss - self.curr_w_topic_loss) + ask_question_loss * self.args.w_question_loss + topic_loss * self.curr_w_topic_loss
                
                valid_losses.append(loss.detach())
                valid_lm_losses.append(lm_loss.detach())
                valid_question_losses.append(ask_question_loss.detach())
                valid_topic_losses.append(topic_loss.detach())
                ppl = torch.exp(lm_loss.detach())
                valid_ppls.append(ppl)
            
            valid_losses = [loss.item() for loss in valid_losses]
            valid_lm_losses = [loss.item() for loss in valid_lm_losses]
            valid_question_losses = [loss.item() for loss in valid_question_losses]
            valid_topic_losses = [loss.item() for loss in valid_topic_losses]

            valid_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls]
            valid_loss = np.mean(valid_losses)
            valid_lm_loss = np.mean(valid_lm_losses)
            valid_question_loss = np.mean(valid_question_losses)
            valid_topic_loss = np.mean(valid_topic_losses)
            valid_ppl = np.mean(valid_ppls)
            
            if math.isnan(valid_ppl):
                valid_ppl = 1e+8
              
        return valid_loss, valid_ppl, valid_lm_loss, valid_question_loss, valid_topic_loss
        
              
    def infer(self):
        print("Let's start!")
        print(f"If you want to quit the conversation, please type \"{self.args.end_command}\".")
        self.model.eval()
        self.fix_seed(self.args.seed)
        
        with torch.no_grad():
            input_hists = []
            
            while True:
                utter = input("You: ")
                if utter == self.args.end_command:
                    print("Bot: Good bye.")
                    break
                
                input_ids = [self.args.sp1_id] + self.tokenizer.encode(utter)
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
                
                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)                
                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                print(f"Bot: {res}")
                input_hists.append([self.args.sp2_id] + self.tokenizer.encode(res))
                
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
    
    def compute_question_loss(self, logits, ask_question_labels):
        token_seq = torch.argmax(logits, dim=-1)
        last_token = token_seq[:, -1]
        predicted_ask_question = []
        for utt in range(last_token.size(0)):
            token_id = last_token[utt].item()
            if token_id != 30:
                predicted_ask_question.append(torch.LongTensor([0]))
            else:
                predicted_ask_question.append(ask_question_labels[utt].view(1))
        predicted_ask_question = torch.cat(predicted_ask_question)
        predicted_ask_question = F.one_hot(predicted_ask_question, 3)
        criteria = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.05, 0.4, 0.55]).to(self.args.device), ignore_index=self.args.pad_id)
        ask_question_loss = criteria(predicted_ask_question.float().to(self.args.device), ask_question_labels)
        return ask_question_loss

    def compute_topic_loss(self, logits, labels):
        token_seq = torch.argmax(logits, dim=-1)
        pred_res = self.tokenizer.batch_decode(token_seq, skip_special_tokens=True)
        gt_res = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        if self.args.topic_model == "LDA":
            pred_res =  [self.id2word.doc2bow(res.lower().split()) for res in pred_res]
            gt_res =  [self.id2word.doc2bow(res.lower().split()) for res in gt_res]
            pred_res_topic, _ = self.topic_model.inference(pred_res)
            pred_res_topic = F.softmax(torch.from_numpy(pred_res_topic), dim=-1)
            gt_res_topic, _ = self.topic_model.inference(gt_res)
            gt_res_topic = F.softmax(torch.from_numpy(gt_res_topic), dim=-1)
        elif self.args.topic_model == "BERT":
            _, pred_res_topic = self.topic_model.transform(pred_res) 
            _, gt_res_topic = self.topic_model.transform(gt_res)

        pred_res_topic, gt_res_topic = torch.tensor(pred_res_topic, dtype=torch.float), torch.tensor(gt_res_topic, dtype=torch.float)
        criteria = nn.KLDivLoss(reduction="batchmean", log_target=True)
        topic_loss = criteria(pred_res_topic, gt_res_topic)
        return topic_loss
        
    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
                    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--mode', type=str, required=True, help="The running mode: train or inference?")
    parser.add_argument('--data_dir', type=str, default="data", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of the validation data files' name.")
    parser.add_argument('--model_type', type=str, default="gpt2", help="The model type of GPT-2.")
    parser.add_argument('--bos_token', type=str, default="<bos>", help="The BOS token.")
    parser.add_argument('--pad_token', type=str, default="<pad>", help="The PAD token.")
    parser.add_argument('--sp1_token', type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument('--sp2_token', type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    parser.add_argument('--lr', type=float, default=2e-5, help="The learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--batch_size', type=int, default=8, help="The batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--num_epochs', type=int, default=10, help="The number of total epochs.")
    parser.add_argument('--train_topic_epochs', type=int, default=5, help="The number of epochs apdating topic loss.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument('--max_turns', type=int, default=5, help="The maximum number of dialogue histories to include.")
    parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--w_question_loss', type=float, default=0.25, help="The weight value of question loss.")
    parser.add_argument('--w_topic_loss', type=float, default=0.25, help="The weight value of topic loss.")
    parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--ckpt_dir', type=str, default="saved_models", help="The directory name for saved checkpoints.")
    parser.add_argument('--exp_name', type=str, default="latest", required=False, help="The name of experiment name")
    parser.add_argument('--topic_model', type=str, default='LDA', help="The type of topic model.")
    parser.add_argument('--topic_model_ckpt_path', type=str, default='saved_models/topic_modelling/mymodel', help="The path of the trained checkpoint of the topic model.")
    parser.add_argument('--end_command', type=str, default="Abort!", help="The command to stop the conversation when inferencing.")
              
    args = parser.parse_args()
    
    assert args.mode in ["train", "infer"]
    assert args.model_type in [
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        "microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"
    ]
    
    args.data_dir = f"{args.data_dir}/{args.model_type}"
    args.ckpt_dir = f"{args.ckpt_dir}/{args.model_type}/{args.exp_name}"
              
    if args.mode == 'train':
        manager = Manager(args)
        manager.train()        
    elif args.mode == 'infer':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
    
        manager = Manager(args)
        manager.infer()
