#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import time

class Config:
    def __init__(self):
        pass

    def get(self, key, default_value=None):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return default_value

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__.get(name, None)

from accelerate import load_checkpoint_and_dispatch, dispatch_model
from accelerate.utils import get_balanced_memory, infer_auto_device_map, set_module_tensor_to_device
from accelerate import init_empty_weights
class PlayGround:
    def __init__(self, 
                 ckpt_path,
                 config_path,
                 tokenizer_path,
                 add_sep=False,
                 multi_gpu=False,
                 use_fast=False,
                 model_num=0
                 ):
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.add_sep = add_sep
        self.tokenizer = self.make_tokenizer(tokenizer_path, use_fast)
        self.multi_gpu = multi_gpu
        self.model_num = model_num
        self.model = self.make_model()
        self.trial_num = 1

    def make_model(self):
        print("Start loading HF Model")
        from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
        if self.multi_gpu:
            model = LlamaForCausalLM.from_pretrained(self.ckpt_path, device_map='auto')
            model.eval()
        else:
            config_dict = AutoConfig.from_pretrained(self.config_path)
            model = AutoModelForCausalLM.from_pretrained(self.ckpt_path)
            # model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.cuda(self.model_num)
            
        model.half()
        print("Finish loading model from %s" % self.ckpt_path)
        return model

    def make_tokenizer(self, tokenizer_path, use_fast=False):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_len=-1, use_fast=use_fast)
        return tokenizer

    def preprocess_text(self, text, add_sep):
        ids = self.tokenizer.encode(text)
        if add_sep:
            sep_id = self.tokenizer.sep_token_id
            if sep_id is None:         
                sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
            ids = ids + [sep_id]
        prompt_len = len(ids)
        return torch.tensor(ids).unsqueeze(0).to(torch.long).cuda(self.model_num), prompt_len

    @torch.no_grad()
    def generate(self, text):
        input_ids, prompt_len = self.preprocess_text(text, self.add_sep)
        time1 = time.perf_counter()
        out = self.model.generate(input_ids=input_ids,
                                    max_length=2048,
                                    do_sample=False,
                                    num_return_sequences=self.trial_num)
        print("delta time:", time.perf_counter() - time1)
        sequence = out[0].cpu().numpy().tolist()
        completion = self.tokenizer.decode(sequence[prompt_len:])
        return completion
    
    def generate_sample(self, text):
        input_ids, prompt_len = self.preprocess_text(text, self.add_sep)
        time1 = time.perf_counter()
        out = self.model.generate(input_ids=input_ids,
                                    max_length=2048,
                                    do_sample=True,
                                    top_p=0.7,
                                    temperature=1.0,
                                    num_return_sequences=self.trial_num)
        print("delta time:", time.perf_counter() - time1)
        sequence = out[0].cpu().numpy().tolist()
        completion = self.tokenizer.decode(sequence[prompt_len:])
        return completion
    
    @torch.no_grad()
    def generate_score(self, text, token_num, sequence_num):
        input_ids, _ = self.preprocess_text(text, self.add_sep)
        time1 = time.perf_counter()
        out = self.model.generate(input_ids=input_ids,
                                    max_new_tokens=3,
                                    do_sample=False,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    num_return_sequences=self.trial_num)
        print("delta time:", time.perf_counter() - time1)
        return torch.nn.functional.softmax(out.scores[sequence_num][0], dim=0)[token_num].item()
    
    @torch.no_grad()
    def generate_action(self, text, sequence_num, threshold=0):
        input_ids, _ = self.preprocess_text(text, self.add_sep)
        time1 = time.perf_counter()
        out = self.model.generate(input_ids=input_ids,
                                    max_new_tokens=3,
                                    do_sample=False,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    num_return_sequences=self.trial_num)
        print("delta time:", time.perf_counter() - time1)
        res = [
            [torch.nn.functional.softmax(out.scores[sequence_num][0], dim=0)[2008].item() + threshold, " [SeekAdvice]\n"],
            [torch.nn.functional.softmax(out.scores[sequence_num][0], dim=0)[7974].item(), " [SearchProduct]\n"],
            [torch.nn.functional.softmax(out.scores[sequence_num][0], dim=0)[23084].item(), " [PredictAnswer]\n"],
        ]
        res = sorted(res, key=lambda x: x[0], reverse=True)
        return res[0][1]
