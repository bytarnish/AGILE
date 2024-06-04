import argparse

import json
import time
from tqdm import tqdm
from prettytable import PrettyTable
import os

import sys
sys.path.append("..")
from models import AzureClient
from utils import load_prompt, dump_jsonline, load_jsonl_in_json


def chunks(lst, n):
    n = max(1, n)
    return [lst[i:i+n] for i in range(0, len(lst), n)]

class FactQAGenerator(object):
    def __init__(self, args):
        self.client = AzureClient(api_key="")

        # === load arguments ===
        self.debug_mode = True if args.debug == "yes" else False

        self.category = args.category 
        self.group = args.group 

        # set other args
        if self.debug_mode:
            self.review_chunk_size = 3
        else:
            self.review_chunk_size = 30
        self.raw_qa_ratio = 2 / 3

        # set paths
        self.meta_data_path = f'./qa_data/meta_data/{self.category}/group_meta.json'
        self.raw_category_path = f"./qa_data/raw_data/fact_qa/{self.category}"
        if self.debug_mode:
            self.raw_category_path = f"./qa_data/raw_data/fact_qa_debug/{self.category}"
        self.raw_group_path = self.raw_category_path + f"/{self.group}"
        self.output_category_path = f"./qa_data/product_data/fact_qa/{self.category}"
        if self.debug_mode:
            self.output_category_path = f"./qa_data/product_data/fact_qa_debug/{self.category}"
        self.output_group_path = self.output_category_path + f"/{self.group}"

        self.check_path_existance()
        self.get_group_meta_data()


    
    def generate_product_qa(self, reviews, meta, raw_qa_batch_size):
        prompt = load_prompt('../prompt/fact_qa_generate_from_reviews').format(raw_qa_batch_size=raw_qa_batch_size) + '\n'
        prompt = prompt + "--- metadata ---\n"
        prompt = prompt + meta
        prompt = prompt + "--- reviews ---\n"
        prompt = prompt + reviews
        result = self.client(prompt, model='gpt-4-1106-preview', max_tokens=2000)
        return result

    def filter_question(self, text):
        prompt = load_prompt('../prompt/fact_qa_natural_filter')
        prompt = prompt + text
        result = self.client(prompt, model='gpt-4-1106-preview', max_tokens=2)
        if (('Yes' in result) or ('yes' in result)):
            return True
        else:
            return False

    def filter_answer(self, question, answer):
        prompt = load_prompt('../prompt/fact_qa_answer_filter') + '\n'
        prompt = prompt + "--- question ---\n"
        prompt = prompt + question + '\n'
        prompt = prompt + "--- answer ---\n"
        prompt = prompt + answer + '\n'
        result = self.client(prompt, model='gpt-4-1106-preview', max_tokens=2)
        if (('Yes' in result) or ('yes' in result)):
            return True
        else:
            return False

    def filter_duplication(self, question, q_occur):
        if question not in q_occur:
            q_occur[question] = 1 
            return True
        else:
            return False

    def modify_qa_pairs(self, raw_qa):
        prompt = load_prompt('../prompt/fact_qa_modify') + '\n'
        prompt = prompt + "[question]\n"
        prompt = prompt + raw_qa['q'] + '\n'
        prompt = prompt + "[answer]\n"
        prompt = prompt + raw_qa['a'] + '\n'
        result = self.client(prompt, model='gpt-4-1106-preview', max_tokens=200)
        result = [x for x in result.split('\n') if x != '']
        if len(result) == 3 and result[0][:10] == "question: " and result[1][:13] == "long_answer: " and result[2][:14] == "short_answer: ":
            qa = {
                "question": result[0][10:],
                "long_answer": result[1][13:],
                "short_answer": result[2][14:]
            }
            return qa
        else:
            return None

    def format_qa_pairs(self, qa, id, asin):
        new_qa = {
            "id": 'fact_qa_%s'%id, 
            "asin": asin, 
            "question": qa['question'],
            "long_answer": qa["long_answer"], 
            "short_answer": qa["short_answer"], 
            "type": "fact_qa", 
            "extra_data": None, 
        }
        return new_qa

    # check path existance
    def check_path_existance(self):
        for path in [self.raw_category_path, self.raw_group_path, self.output_category_path, self.output_group_path]:
            if not os.path.exists(path):
                os.mkdir(path)


    def get_group_meta_data(self):
        with open(self.meta_data_path, 'r') as fin:
            meta_data = json.load(fin)[self.group]

        # get reviews and metas
        self.reviews = {}
        self.metas = {}
        for product in meta_data:
            asin = product['asin']
            if self.debug_mode:
                self.reviews[asin] = product['review'][:5]
            else:
                self.reviews[asin] = product['review']
            tmp = {k:v for k, v in product.items() if k != 'review'}
            self.metas[asin] = tmp

        # get group asin_list
        self.asin_list = [asin for asin in self.metas.keys()]
        if self.debug_mode:
            self.asin_list = self.asin_list[:2]

    # generate raw qa pairs
    def generate_raw_qa(self):
        for asin in tqdm(self.asin_list, desc="Fact QA Generating"):
            with open(self.raw_group_path +  f'/raw_qa_{asin}', 'w') as fout:
                # split reviews
                review_chunks = chunks(self.reviews[asin], self.review_chunk_size) # not real iterator, use yield if want iterator
                loop = tqdm(enumerate(review_chunks), total=len(review_chunks), desc="Raw QA Generating")
                start_time = time.time()
                for t_idx, chunk in loop:
                    review_str = ''.join([str(item_idx) + '. ' + item + '\n' for item_idx, item in enumerate(chunk)])
                    qa = self.generate_product_qa(review_str, json.dumps(self.metas[asin]), int(self.raw_qa_ratio * len(chunk)))
                    fout.write(qa + '\n')


    # modify qa pairs
    def modify_qa(self):
        for asin in tqdm(self.asin_list, desc="Fact QA Generating"):
            raw_qa_pairs = []
            # load raw qa pairs
            with open(self.raw_group_path +  f'/raw_qa_{asin}', 'r') as fin:
                line = fin.readline()
                question = ''
                answer = ''
                while line:
                    if line[0] == '*':
                        if question != '' and answer != '':
                            raw_qa_pairs.append({'q': question, 'a': answer})
                        question = fin.readline()[2:]
                        answer = fin.readline()[2:]
                    line = fin.readline()
                if question != '' and answer != '':
                    raw_qa_pairs.append({'q': question, 'a': answer})
            # modify qa pairs
            qa_pairs = []
            q_occur = {}
            # polish qa pairs; get short answer
            with open(self.raw_group_path + f'/mod_{asin}', 'w') as fout:
                loop = tqdm(enumerate(raw_qa_pairs), total=len(raw_qa_pairs), desc=f"Modifing {asin}")
                start_time = time.time()
                for t_idx, raw_qa in loop:
                    # modify qa
                    qa = self.modify_qa_pairs(raw_qa)
                    if qa:
                        fout.write(json.dumps({'prev': raw_qa, 'cur': qa}) + '\n')
                        fout.flush()
                    else:
                        # print('*** Format error ***')
                        continue
                    # filter question duplication
                    valid = self.filter_duplication(qa['question'], q_occur)
                    if valid:
                        qa_pairs.append(qa)
            # save mod result
            with open(self.raw_group_path + f'/modified_{asin}.json', 'w') as fout:
                for qa in qa_pairs:
                    fout.write(json.dumps(qa) + '\n')

    # filter qa pairs
    def filter_qa(self):
        for asin in tqdm(self.asin_list, desc="Fact QA Generating"):
            # get modified qa pairs
            qa_pairs = load_jsonl_in_json(self.raw_group_path + f'/modified_{asin}.json')
            
            retain_qa_pairs = []
            drop_qa_pairs = []
            loop = tqdm(enumerate(qa_pairs), total=len(qa_pairs), desc=f"Filtering {asin}")
            start_time = time.time()
            for t_idx, qa in loop:
                question, long_answer = qa['question'], qa['long_answer']
                # filter question
                valid_1 = self.filter_question(qa['question'])
                # filter answer
                valid_2 = self.filter_answer(qa['question'], qa['long_answer'])
                
                if valid_1 and valid_2:
                    retain_qa_pairs.append({'question': qa['question'], 'long_answer': qa['long_answer'], 'short_answer': qa['short_answer']})
                else:
                    drop_qa_pairs.append({'question': qa['question'], 'long_answer': qa['long_answer'], 'short_answer': qa['short_answer']})
            
            # save qa pairs
            with open(self.raw_group_path + f'/filter_qa_{asin}.json', 'w') as fout:
                json.dump(retain_qa_pairs, fout)
            with open(self.raw_group_path + f'/dropped_qa_{asin}.json', 'w') as fout:
                json.dump(drop_qa_pairs, fout)



    def collect_qa(self):
        qa_type_dict = {asin: {'bool': 0, 'single<4word': 0, 'single>=4word': 0, 'multiple': 0, 'total': 0} for asin in self.asin_list}
        final_qa_list =[]
        multiple_qa_list = []
        single_long_list = []
        for asin in self.asin_list:
            with open(self.raw_group_path + f'/filter_qa_{asin}.json', "r") as fin:
                data = json.load(fin)
            for item in data:
                qa_type_dict[asin]['total'] += 1
                if item['short_answer'] == 'yes' or item['short_answer'] == 'no':
                    qa_type_dict[asin]['bool'] += 1
                    final_qa_list.append(self.format_qa_pairs(item, len(final_qa_list), asin))
                elif ',' in item['short_answer']:
                    qa_type_dict[asin]['multiple'] += 1
                    multiple_qa_list.append(self.format_qa_pairs(item, len(multiple_qa_list), asin))
                else:
                    if len(item['short_answer'].split(' ')) < 4:
                        qa_type_dict[asin]['single<4word'] += 1
                        final_qa_list.append(self.format_qa_pairs(item, len(final_qa_list), asin))
                    else:
                        qa_type_dict[asin]['single>=4word'] += 1
                        single_long_list.append(self.format_qa_pairs(item, len(single_long_list), asin))

        if self.debug_mode:
            dump_jsonline(self.output_group_path + "/debug_fact_qa_full.jsonl", final_qa_list)
            dump_jsonline(self.output_group_path + "/debug_fact_qa_multiple.jsonl", multiple_qa_list)
            dump_jsonline(self.output_group_path + "/debug_fact_qa_single_long.jsonl", single_long_list)
        else:
            dump_jsonline(self.output_group_path + "/product_fact_qa_full.jsonl", final_qa_list)
            dump_jsonline(self.output_group_path + "/product_fact_qa_multiple.jsonl", multiple_qa_list)
            dump_jsonline(self.output_group_path + "/product_fact_qa_single_long.jsonl", single_long_list)

        # statistics output
        table = PrettyTable()

        table.field_names = ["Asin", "Bool", "Single<4word", "Single>=4word", "Multiple", "Total"]
        for asin in qa_type_dict:
            table.add_row([asin, qa_type_dict[asin]['bool'], qa_type_dict[asin]['single<4word'], qa_type_dict[asin]['single>=4word'], qa_type_dict[asin]['multiple'], qa_type_dict[asin]['total']])
        table.add_row([
                'Sum', 
                sum([qa_type_dict[asin]['bool'] for asin in qa_type_dict]), 
                sum([qa_type_dict[asin]['single<4word'] for asin in qa_type_dict]), 
                sum([qa_type_dict[asin]['single>=4word'] for asin in qa_type_dict]), 
                sum([qa_type_dict[asin]['multiple'] for asin in qa_type_dict]), 
                sum([qa_type_dict[asin]['total'] for asin in qa_type_dict])])
        print(table)
        if self.debug_mode:
            with open(self.output_group_path +  "/debug_statistics.txt", "w") as fin:
                fin.write(str(table))
        else:
            with open(self.output_group_path +  "/statistics.txt", "w") as fin:
                fin.write(str(table))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='no')
    parser.add_argument('--group', type=str, default='no')
    parser.add_argument('--debug', type=str, default='no')
    args = parser.parse_args()

    fact_qa_generator = FactQAGenerator(args)
    fact_qa_generator.generate_raw_qa()
    fact_qa_generator.modify_qa()
    fact_qa_generator.filter_qa()
    fact_qa_generator.collect_qa()