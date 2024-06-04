import os
import json
import argparse
from client import SQLClient
from generate import PlayGround
import random
from models import (
    AzureClient,
    # call_search_azure,
    # call_vicuna
)
from utils import (
    load_prompt,
    parse_answer,
    get_action,
    get_sql,
    eval_search,
    eval_predict,
    get_schema_description,
    text2emb,
    search_k_memory,
    eval_predict_long,
    gen_hash_id,
    search_past_similar
)

class Agent:
    def __init__(
        self,
        group,
        category,
        model="gpt-4",
        reflection=True,
        seek_advice=True,
        use_memory=True,
        model_file="",
        agent_gpu="cuda:0",
        seek_advice_budget_file="",
        extra="1",
        agent_prompt="",
        agent_reflection_prompt="",
        do_sample=False,
        similarity_thred=0.46,
        azure_key=None,
    ):
        self.group = group
        self.category = category
        self.model = model
        self.reflection = reflection
        self.seek_advice = seek_advice
        self.use_memory = use_memory
        self.do_sample = do_sample
        self.similarity_thred = similarity_thred
        self.agent_prompt = load_prompt(agent_prompt)
        self.agent_reflection_prompt = load_prompt(agent_reflection_prompt)
        self.azure_client = AzureClient(api_key=azure_key)

        try:
            self.extra = int(extra)
            if self.extra < 0:
                self.extra = -1
        except:
            self.extra = -1
        print("Retrieve {} resultes, -1 for filling 512 tokens".format(self.extra))

        try:
            self.seek_advice_budget = json.loads(seek_advice_budget_file)
        except:
            print("seek_advice_budget_file load err, set budget zero. Only effective when seek_advice is False")
            self.seek_advice_budget = {}
        
        if model_file != "":
            self.playground = PlayGround(
                ckpt_path=model_file, 
                config_path=os.path.join(model_file, "config.json"),
                tokenizer_path=model_file,
                model_num=agent_gpu
            )
            self.tokenizer = self.playground.tokenizer
        else:
            self.playground = None
            self.tokenizer = None
    

    def reset_storage(self):
        self.history = {}
        self.history_emb = {}
        self.history_question_emb = {}
        self.memory = []
        self.memory_emb = []
        self.total = {}
    

    def call_model(self, prompt, model, max_token=1000):
        # if model == 'vicuna':
        #     pass
        if model == 'vicuna-sft':
            assert self.playground is not None, "vicuna-sft model must be specified with model_file"
            if self.do_sample:
                resp = self.playground.generate_sample(prompt).strip('\n').strip('response:').strip(' ').strip('<s>').strip('</s>')
            else:
                resp = self.playground.generate(prompt).strip('\n').strip('response:').strip(' ').strip('<s>').strip('</s>')
        else:
            resp = self.azure_client(prompt, model, max_token)
        return resp


    def get_memory(self, question, asin):
        # retrieve memory and history
        if self.extra >= 0:
            context_knowledge = search_k_memory(self.memory, self.memory_emb, question, tokenizer=self.tokenizer, target_num=self.extra)
            context_history = search_k_memory(self.history[asin], self.history_emb[asin], question, tokenizer=self.tokenizer, target_num=self.extra)
        else:
            context_knowledge = search_k_memory(self.memory, self.memory_emb, question, tokenizer=self.tokenizer)
            context_history = search_k_memory(self.history[asin], self.history_emb[asin], question, tokenizer=self.tokenizer)
        return context_knowledge, context_history
    

    def update_memory(self, data, asin, long_answer):
        question = data["question"]
        qa_pair = 'Question: ' + question + '\nAnswer: ' + str(long_answer) + '\n'
        if qa_pair not in self.history[asin]:
            self.history[asin].append('Question: ' + question + '\nAnswer: ' + str(long_answer) + '\n')
            self.history_question_emb[asin].append(text2emb(question))
            self.history_emb[asin].append(text2emb(self.history[asin][-1]))

        if self.reflection:
            reflection = None
            prompt = self.agent_reflection_prompt.format(product_category=self.group, question=question, answer=long_answer)
            if self.cache_data is not None:
                hash_id = gen_hash_id(prompt)
                if hash_id in self.cache_data:
                    reflection = self.cache_data[hash_id]
            #     else:
            #         reflection = self.call_model(prompt, model=self.model)
            # else:
            #     reflection = self.call_model(prompt, model=self.model)
            data['reflection_prompt'] = prompt
            data['reflection_resp'] = reflection
            if reflection is not None and 'no information' not in reflection.lower() and reflection not in self.memory:
                self.memory.append(reflection.strip() + '\n')
                self.memory_emb.append(text2emb(self.memory[-1]))


class MedicalAgent(Agent):
    def __init__(
        self,
        category,
        group="medical",
        model="gpt-4",
        reflection=True,
        seek_advice=True,
        use_memory=True,
        model_file="",
        agent_gpu="cuda:0",
        seek_advice_budget_file="",
        extra="1",
        agent_prompt="prompt/agent_for_med_ppo",
        agent_reflection_prompt="prompt/agent_reflection_med",
        reflection_cache_file="",
        do_sample=False,
        similarity_thred=0.46,
        azure_key=None,
    ):
        super().__init__(group, category, model, reflection, seek_advice, use_memory, model_file, agent_gpu, seek_advice_budget_file, extra, agent_prompt, agent_reflection_prompt, do_sample, similarity_thred, azure_key)
        
        if reflection_cache_file != "":
            with open(reflection_cache_file) as fin:
                cache_data = [json.loads(line) for line in fin]
                self.cache_data = {gen_hash_id(self.agent_reflection_prompt.format(product_category=self.group, question=data["question"], answer=data["options"][data["answer"]])):data['reflection'] for data in cache_data}
        else:
            self.cache_data = None


    def reset_storage(self):
        super().reset_storage()
        self.history[self.category] = []
        self.history_emb[self.category] = []
        self.history_question_emb[self.category] = []
        self.total[self.category] = 0

    
    def next_action(self, data, asin=""):
        has_seek_advice = False

        # Determine if seek advice is required
        if not self.seek_advice:
            if self.total[asin] < self.seek_advice_budget.get(asin, 0):
                # top k force seek advice and budget available
                has_seek_advice = True
                resp = ' [SeekAdvice]\n'
                action = " [SeekAdvice]\n"
            else:
                resp = self.call_model(data['agent_prompt'], self.model)
                action = get_action(resp)
                if "SeekAdvice" in action:
                    assert self.playground is not None, "no SeekAdvice budget, change prompt!"                        
                    action = " [PredictAnswer]\n"
                    cot = resp.split("[SeekAdvice]")[0]
                    prefix = cot + '\n' + action.strip() + "\n[Answer]:"
                    resp = prefix + self.call_model(data['agent_prompt'] + prefix, self.model)
        else:
            resp = self.call_model(data['agent_prompt'], self.model)
            action = get_action(resp)
            if "SeekAdvice" in action:
                has_seek_advice = True

        return action, resp, has_seek_advice


    def parse_answer(self, data, has_seek_advice, action):
        
        if "PredictAnswer" in action:
            data['resp_short_answer'] = parse_answer(data['agent_output'], '[Answer]').strip()
            if data["resp_short_answer"] in data["options"]:
                data["resp_long_answer"] = data["options"][data["resp_short_answer"]]
            else:
                data["resp_long_answer"] = data["resp_short_answer"]

        if "SeekAdvice" in action:
            data['resp_long_answer'] = data['resp_short_answer'] = '[SeekAdvice]'
        
        data["action"] = action.strip()
        return data, has_seek_advice

    
    def step_one(self, data):
        asin = self.category
        question = data['question']
        
        context_knowledge, context_history = self.get_memory(question, asin)
        option = "\n".join([k + ": " + v for k, v in data["options"].items()])

        qa_similar_num, memory_similar_num = search_past_similar(
            question,
            self.history_question_emb[asin],
            self.memory_emb,
            qa_thred=self.similarity_thred,
            memory_thred=self.similarity_thred
        )

        # prompt agent and predict the next action
        agent_prompt = self.agent_prompt.format(
            question=data["question"] + '\n\n' + option,
            knowledge=context_knowledge,
            history=context_history,
            similar_past_question_num=qa_similar_num,
            similar_past_knowledge_num=memory_similar_num,
            round=data['rounds'],
        )
        data['agent_prompt'] = agent_prompt
        action, resp, has_seek_advice = self.next_action(data, asin)
        data['agent_output'] = resp
        data, has_seek_advice = self.parse_answer(data, has_seek_advice, action)
        self.total[asin] += 1

        # seek advice and reflection
        if has_seek_advice and self.use_memory:
            self.update_memory(data, self.category, data["options"][data["answer"]])
        data["memory"] = self.memory
        data["history"] = self.history[asin]
        return data
    

    def eval(self, data, *_):
        if data["action"] == "[SeekAdvice]":
            data['short_answer_eval'] = True
        else:
            data['short_answer_eval'] = data["resp_short_answer"].lower() == data["answer"].lower()
        return data
    

class ProductAgent(Agent):
    def __init__(
        self,
        group,
        category,
        model="gpt-4",
        reflection=True,
        seek_advice=True,
        use_memory=True,
        model_file="",
        agent_gpu="cuda:0",
        seek_advice_budget_file="",
        extra="1",
        agent_prompt="agile/prompt/agent_for_product",
        agent_reflection_prompt="agile/prompt/agent_reflection_output",
        reflection_cache_file="",
        do_sample=False,
        generate_short_answer_prompt="agile/prompt/agent_generate_short_answer",
        similarity_thred=0.46,
        azure_key=None,
        db_path="agile/product_qa.db",
    ):
        super().__init__(group, category, model, reflection, seek_advice, use_memory, model_file, agent_gpu, seek_advice_budget_file, extra, agent_prompt, agent_reflection_prompt, do_sample, similarity_thred, azure_key)
        self.schema = json.load(open(f"data/test/{self.group}/schema.json"))
        self.group_metadata = json.load(open(f"data/test/{self.group}/metadata.json"))
        self.all_products = [{"asin": k, "title": v['title']} for k, v in self.group_metadata.items()]
        self.schema_des = get_schema_description(self.schema)
        self.generate_short_answer_prompt = load_prompt(generate_short_answer_prompt)
        self.sql_client = SQLClient(db_path)
        for key in self.schema:
            if 'unit' in self.schema[key]:
                for asin in self.group_metadata:
                    self.group_metadata[asin][key] = self.group_metadata[asin][key] + ' ' + self.schema[key]['unit']
        
        if reflection_cache_file != "":
            with open(reflection_cache_file) as fin:
                cache_data = [json.loads(line) for line in fin]
                self.cache_data = {data['prompt_hash_id']:data['completion'] for data in cache_data}
        else:
            self.cache_data = None

    def reset_storage(self):
        super().reset_storage()
        for product in self.all_products:
            self.history[product["asin"]] = []
            self.history_emb[product["asin"]] = []
            self.history_question_emb[product["asin"]] = []
            self.total[product["asin"]] = 0


    def generate_short_answer(self, question, answer):
        prompt = self.generate_short_answer_prompt.format(question=question, answer=answer)
        resp = self.call_model(prompt, model=self.model, max_token=50)
        short_answer = resp.strip('.\n,!')
        return short_answer


    def parse_answer(self, data, has_seek_advice, action):
        # take action and parse the answer
        if "SearchProduct" in action:
            s, success = get_sql(data['agent_output']) #parse sql
            if success:
                asin_list = self.sql_client.sql_query(s, self.group)
                data['sql_execution'] = asin_list
                if asin_list == "err":
                    success = False
                else:
                    data['resp_long_answer'] = data['resp_short_answer'] = [str(asin) for asin in asin_list]
            if not success:
                if not self.seek_advice:
                    # call api error, return empty list
                    data['resp_long_answer'] = data['resp_short_answer'] = []
                else:
                    has_seek_advice = True
                    action = " [SeekAdvice]\n"
        
        if "PredictAnswer" in action:
            data['resp_long_answer'] = parse_answer(data['agent_output'], '[Answer]')
            if self.model == 'vicuna-sft':
                data['resp_short_answer'] = parse_answer(data['agent_output'], '[Short Answer]')
            else:
                data['resp_short_answer'] = self.generate_short_answer(data['question'], data['agent_output'])

        if "SeekAdvice" in action:
            data['resp_long_answer'] = data['resp_short_answer'] = '[SeekAdvice]'
        
        data["action"] = action.strip()
        return data, has_seek_advice

    
    def step_one(self, data):
        asin = data['asin']
        question = data['question']
        
        context_knowledge, context_history = self.get_memory(question, asin)
        qa_similar_num, memory_similar_num = search_past_similar(
            question,
            self.history_question_emb[asin],
            self.memory_emb,
            qa_thred=self.similarity_thred,
            memory_thred=self.similarity_thred
        )

        # prompt agent and predict the next action
        agent_prompt = self.agent_prompt.format(
            schema=self.schema_des,
            asin=asin,
            question=question,
            product_category=self.group,
            metadata=json.dumps(self.group_metadata[asin]),
            knowledge=context_knowledge,
            history=context_history,
            similar_past_question_num=qa_similar_num,
            similar_past_knowledge_num=memory_similar_num,
            round=data['rounds'],
        )
        data['agent_prompt'] = agent_prompt
        action, resp, has_seek_advice = self.next_action(data, asin)
        data['agent_output'] = resp
        data, has_seek_advice = self.parse_answer(data, has_seek_advice, action)
        self.total[asin] += 1

        # seek advice and reflection
        if has_seek_advice and self.use_memory:
            self.update_memory(data, asin, data["long_answer"])
        data["memory"] = self.memory
        data["history"] = self.history[asin]
        return data
    
    def next_action(self, data, asin=""):
        has_seek_advice = False
        resp = None

        # Determine if seek advice is required
        if not self.seek_advice:
            if self.total[asin] < self.seek_advice_budget.get(asin, 0):
                # top k force seek advice and budget available
                has_seek_advice = True
        else:
            resp = self.call_model(data['agent_prompt'], self.model)
            action = get_action(resp)
            if "SeekAdvice" in action:
                has_seek_advice = True

        # generate action and response
        if resp is None:
            if has_seek_advice:
                resp = ' [SeekAdvice]\n'
                action = " [SeekAdvice]\n"
            else:
                resp = self.call_model(data['agent_prompt'], self.model)
                action = get_action(resp)
                if "SeekAdvice" in action:
                    assert self.playground is not None, "no SeekAdvice budget, change prompt!"                        
                    pred_score = self.playground.generate_score(data['agent_prompt'], token_num=23084, sequence_num=2)
                    search_score = self.playground.generate_score(data['agent_prompt'], token_num=7974, sequence_num=2)
                    if pred_score > search_score:
                        action = " [PredictAnswer]\n"
                        resp = action + "[Answer]:" + self.call_model(data['agent_prompt'] + action + "[Answer]:", self.model)
                    else:
                        action = " [SearchProduct]\n"
                        resp = action + self.call_model(data['agent_prompt'] + action, self.model)
        return action, resp, has_seek_advice

    def eval(self, data, eval_long, prompt_file):
        if data["action"] == "[SeekAdvice]":
            data['long_answer_eval'] = data['short_answer_eval'] = True
            data['long_answer_eval_reason'] = 'seek_advice, correct answer'
        elif data['action'] == "[SearchProduct]":
            data['long_answer_eval'] = data['short_answer_eval'] = eval_search(data['resp_short_answer'], data['short_answer'])
            data['long_answer_eval_reason'] = 'SearchProduct, eval according to rules'
        else:
            data["short_answer_eval"] = eval_predict(data['resp_short_answer'], data['short_answer'], self.all_products)
            if isinstance(data['short_answer'], list):
                data['long_answer_eval'], data['long_answer_eval_reason'] = data["short_answer_eval"], "comparison qa, eval short answer"
            elif eval_long:
                data['long_answer_eval'], data['long_answer_eval_reason'] = eval_predict_long(data['question'], data['resp_long_answer'], data['long_answer'], prompt_file)
            else:
                data['long_answer_eval'] = False
                data['long_answer_eval_reason'] = ''
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seek and Reflect Agent')
    parser.add_argument('--group', default="medical")
    parser.add_argument('--category', default="medqa")
    parser.add_argument('--test_file', default="")
    parser.add_argument('--output_file')
    parser.add_argument('--reflection', action="store_true", default=False)
    parser.add_argument('--seek_advice', action="store_true", default=False)
    parser.add_argument('--use_memory', action="store_true", default=False)
    parser.add_argument('--model', default="vicuna-sft") # gpt, vicuna, vicuna-sft
    parser.add_argument('--agent_prompt', default="prompt/agent_for_med")
    parser.add_argument('--model_file', default="") # xxx_hf only for vicuna-sft model
    parser.add_argument('--agent_gpu', default="cuda:0") # not used if not specified
    parser.add_argument('--seek_advice_budget_file', default="")
    parser.add_argument('--extra', default="1") # When specifying a number, it represents the number of retrieve mems and history. Non numbers are full (512 tokens)
    parser.add_argument('--eval_long', action="store_true", default=False) # Whether to verify the effectiveness of long answer during the reasoning process
    parser.add_argument('--eval_prompt_file', default='prompt/auto_evaluate') # GPT-4 long answer evaluation prompt
    parser.add_argument('--reflection_cache_file', default="")
    parser.add_argument('--do_sample', action="store_true", default=False)
    parser.add_argument('--similarity_thred', type=float, default=0.46)
    parser.add_argument('--azure_key', type=str, default="")
    args = parser.parse_args()

    if args.group == "medical":
        agent = MedicalAgent(
            category=args.category,
            model=args.model,
            reflection=args.reflection,
            seek_advice=args.seek_advice,
            use_memory=args.use_memory,
            model_file=args.model_file,
            agent_prompt=args.agent_prompt,
            seek_advice_budget_file=args.seek_advice_budget_file,
            reflection_cache_file="agile/cache/med_ref.jsonl",
            azure_key=args.azure_key,
        )
    else:
        agent = ProductAgent(
            group=args.group,
            category=args.category,
            model=args.model,
            reflection=args.reflection,
            seek_advice=args.seek_advice,
            use_memory=args.use_memory,
            model_file=args.model_file,
            seek_advice_budget_file=args.seek_advice_budget_file,
            agent_prompt=args.agent_prompt,
            reflection_cache_file="agile/cache/productqa_ref.jsonl",
            azure_key=args.azure_key,
        )

    agent.reset_storage()
    with open(args.test_file) as f:
        total_len = len(f.readlines())
    with open(args.test_file) as f, open(args.output_file, 'w') as fw:
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            rounds = str(round(i / total_len * 100, 1)) + "%"
            data["rounds"] = rounds
            data = agent.step_one(data)
            data = agent.eval(data, args.eval_long, args.eval_prompt_file)
            try:
                fw.write(json.dumps(data, ensure_ascii=False) + '\n')
            except:
                del data['sql_execution']
                fw.write(json.dumps(data, ensure_ascii=False) + '\n')

