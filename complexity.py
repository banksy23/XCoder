import json
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM


class Scorer(object):
    def __init__(self, model_name_or_path: str, is_vllm: bool  = False, **kwargs):
        
        self.is_vllm = is_vllm
        
        if not is_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        else:
            
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(model_name_or_path)
            self.sampling_params = SamplingParams(max_tokens = 2, logprobs = 1000)
    def infer_complexity(self, instruction):
        user_input = "You are a helpful assistant. Please identify the complexity score of the following user query. \n##Query: {instruction}  \n##Complexity: ".replace("{instruction}",instruction)
        max_length = 2
        if  self.is_vllm:
            outputs = self.llm.generate(user_input, self.sampling_params)
            
            try:
                logprobs_list = outputs[0].outputs[0].logprobs[0]
            except IndexError:
                return 2.0
        else:
            input_ids = self.tokenizer.encode(user_input, return_tensors = "pt")
            outputs = self.model.generate(input_ids, max_new_tokens = max_length, num_return_sequences = 1, return_dict_in_generate = True, output_scores = True)
            
            try:
                logprobs_list = outputs.scores[0][0]
            except IndexError:
                return 2.0
        score_logits = []
        for k in [16,17,18,19,20,21]:
            try:
                score_logits.append(logprobs_list[k])
            except KeyError:
                return 2.0
        score_logits = np.array(score_logits)
        score_npy = softmax(score_logits, axis=0)
        score_npy = score_npy * np.array([0,1,2,3,4,5])

        score_npy = np.sum(score_npy, axis=0)
        
        return score_npy
            

