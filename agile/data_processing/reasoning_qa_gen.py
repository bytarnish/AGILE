import sys
sys.path.append('..')
import argparse
import json
import os
import random
from models import AzureClient
from utils import get_group_split, load_prompt


class ReasoningQAGenerator(object):
    def __init__(self, group, group_filename):
        self.group = group
        self.group_filename = group_filename
        self.dirname = os.path.join('reasoning_qa', group_filename)
        self.client = AzureClient(api_key='')
        os.makedirs(self.dirname, exist_ok=True)

        with open(os.path.join('amazon_data', group_filename + '.json'), 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        self.meta_data = dict()
        for a in data:
            asin = a['asin']
            self.meta_data[asin] = a
            del self.meta_data[asin]['review']

        with open(os.path.join('..', '..', 'data', get_group_split(group_filename), group_filename, 'schema.json'), 'r', encoding='utf-8') as fin:
            self.features = json.load(fin)
            for key in list(self.features.keys()):
                if key[0].islower():
                    del self.features[key]

    def _evaluate_knowledge(self, knowledge):
        prompt = load_prompt(os.path.join('..', 'prompt', 'evaluate_knowledge_common')).format(knowledge=knowledge)
        ans = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
        if 'Yes' in ans or 'yes' in ans or 'YES' in ans:
            return False

        prompt = load_prompt(os.path.join('..', 'prompt', 'evaluate_knowledge')).format(knowledge=knowledge)
        ans = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
        if 'No' in ans or 'no' in ans or 'NO' in ans:
            return False

        return True

    def _evaluate_question(self, question):
        prompt = load_prompt(os.path.join('..', 'prompt', 'filter_qa_natural')).format(question=question, product_category=self.group)
        ans = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
        if 'No' in ans or 'no' in ans or 'NO' in ans:
            ans1 = False
        else:
            ans1 = True

        prompt = load_prompt(os.path.join('..', 'prompt', 'filter_qa_general')).format(question=question)
        ans = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
        if 'product' in ans.lower():
            ans1 = ans1 and True
        else:
            ans1 = False

        prompt = load_prompt(os.path.join('..', 'prompt', 'filter_qa')).format(question=question)
        ans = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
        if 'experience' in ans:
            ans2 = 'experience'
        else:
            ans2 = 'functionality'
        return ans1, ans2

    def _evaluate_answer_knowledge(self, answer, knowledge):
        prompt = load_prompt(os.path.join('..', 'prompt', 'evaluate_answer_infer_knowledge')).format(info0=answer, info1=knowledge)
        ans = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
        if 'No' in ans or 'no' in ans or 'NO' in ans:
            return False
        return True

    def generate_knowledge(self):
        # generate knowledge piece
        with open(os.path.join(self.dirname, 'knowledge.jsonl'), 'w', encoding='utf-8') as fout:
            for feature_name in self.features:
                for feature_value in self.features[feature_name]['choices']:
                    if 'not applicable' in feature_value:
                        continue
                    print(f'Generating knowledge for {feature_name}: {feature_value} ...')

                    # generate raw knowledge
                    prompt = 'Below is a property of a {product_category} product, please generate as more as domain knowledge about the product related to the property.\n\n'.format(product_category=self.group) + feature_name + ':' + feature_value
                    knowledge = self.client(prompt, model='gpt-4-1106-preview')

                    # generate knowledge piece
                    prompt = load_prompt(os.path.join('..', 'prompt', 'generate_knowledge_piece')).format(product_category=self.group, feature_name=feature_name, feature_value=feature_value, domain_knowledge=knowledge)
                    knowledge_pieces = self.client(prompt, model='gpt-4-1106-preview')

                    # evaluate knowledge piece
                    s = knowledge_pieces.split('\n')
                    for piece in s:
                        if len(piece) == 0:
                            continue
                        idx = piece.find(self.group[:-1])
                        if idx < 0 or idx > 50:
                            continue
                        know = piece[idx:]
                        if not self._evaluate_knowledge(know):
                            continue
                        fout.write(json.dumps({'feature_name': feature_name, 'feature_value': feature_value, 'knowledge': know}, ensure_ascii=False) + '\n')
                        fout.flush()

    def generate_reasoning_qa(self):
        # generate qa pairs
        with open(os.path.join(self.dirname, 'reasoning_qa.jsonl'), 'w', encoding='utf-8') as fout, open(os.path.join(self.dirname, 'knowledge.jsonl'), 'r', encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line)
                feature_name = data['feature_name']
                feature_value = data['feature_value']
                knowledge = data['knowledge']
                print(f'Generating reasoning QA for {feature_name}: {feature_value} ...')

                prompt = load_prompt(os.path.join('..', 'prompt', 'generate_from_knowledge_yes_no')).format(product_category=self.group, feature_name=feature_name, feature_value=feature_value, domain_knowledge=knowledge)
                qa_list_str = self.client(prompt, model='gpt-4-1106-preview')

                qa_list = list()
                s = qa_list_str.split('\n')
                q0 = ''
                a0 = ''
                for line in s:
                    if len(line) == 0:
                        continue
                    if line[0] == 'Q':
                        q0 = line
                    if line[0] == 'A':
                        a0 = line
                        if not self._evaluate_answer_knowledge(a0, knowledge):
                            continue
                        rst1, rst2 = self._evaluate_question(q0)
                        if rst1:
                            qa_list.append({'q': q0, 'a': a0, 'type': rst2})
                fout.write(json.dumps({'feature_name': feature_name, 'feature_value': feature_value, 'knowledge': knowledge, 'qa_list': qa_list}, ensure_ascii=False) + '\n')
                fout.flush()
        os.remove(os.path.join(self.dirname, 'knowledge.jsonl'))

    def generate_product_reasoning_qa(self):
        # generate product qa

        # load meta qa
        qa_data = dict()
        knowledge_all = ''
        with open(os.path.join(self.dirname, 'reasoning_qa.jsonl'), 'r', encoding='utf-8') as fin:
            pre_key = ''
            for line in fin:
                data = json.loads(line)
                feature_name = data['feature_name']
                feature_value = data['feature_value']
                knowledge = data['knowledge']
                knowledge_all += knowledge + '\n'

                key = feature_name + feature_value

                if key != pre_key:
                    qa_data[key] = list()
                qa_data[key].append({'knowledge': knowledge, 'qa_list': [x for x in data['qa_list']]})
                pre_key = key

        def parse_qa(text, word):
            idx = text.lower().find(word)
            if idx == 0:
                new_text = text[len(word):]
                while new_text[0] == ' ' or new_text[0] == ':':
                    new_text = new_text[1:]
                return new_text
            else:
                return text

        # parse qa format
        for key in qa_data:
            for data in qa_data[key]:
                for qa in data['qa_list']:
                    qa['q'] = parse_qa(qa['q'], 'question')
                    qa['a'] = parse_qa(qa['a'], 'answer')

        def make_a_choice(asin, feature_name, feature_list):
            prompt = load_prompt(os.path.join('..', 'prompt', 'auto_fill_feature_table')).format(metadata=self.meta_data[asin], knowledge=knowledge_all, feature_name=feature_name, feature_list=feature_list, product_category=self.group)
            rst = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
            return rst

        def generate_new(text):
            prompt = load_prompt(os.path.join('..', 'prompt', 'generate_new_text')).format(text=text)
            rst = self.client(prompt, model='gpt-4-1106-preview')
            return rst

        with open(os.path.join(self.dirname, 'product_reasoning_qa.jsonl'), 'w', encoding='utf-8') as fout:
            prompt = 'Please summarize the following knowledges for {product_category} products. Only output the summarization and do not output anything else.\n\n'.format(product_category=self.group) + knowledge_all
            summary = self.client(prompt, model='gpt-4-1106-preview')
            knowledge_all = summary
            metadata_table = dict()

            for asin in self.meta_data:
                metadata_table[asin] = dict()
                for feature_name in self.features:
                    print(f'Generating product reasoning QA for {asin} with feature {feature_name} ...')
                    feature_list = '\n'.join(['(' + chr(65 + i) + ') ' + self.features[feature_name]['choices'][i] for i in range(len(self.features[feature_name]['choices']))])

                    j = 0
                    flag = False
                    feature_value = ''
                    while j < 10:
                        rst = make_a_choice(asin, feature_name, feature_list)
                        if len(rst) > 80:
                            j += 1
                            continue
                        for c in rst:
                            if ord(c) >= 65 and ord(c) < 65 + len(self.features[feature_name]['choices']):
                                idx = ord(c) - 65
                                feature_value = self.features[feature_name]['choices'][idx]
                                metadata_table[asin][feature_name] = feature_value
                                flag = True
                                break
                        if flag:
                            break
                        j += 1
                    if not flag:
                        metadata_table[asin][feature_name] = None
                    else:
                        key = feature_name + feature_value
                        if key not in qa_data:
                            continue
                        for data in qa_data[key]:
                            if len(data['qa_list']) == 0:
                                continue
                            knowledge = data['knowledge']
                            idx = int(random.random() * len(data['qa_list']))
                            base_question = data['qa_list'][idx]['q']
                            base_answer = data['qa_list'][idx]['a']
                            type = data['qa_list'][idx]['type']
                            new_question = generate_new(base_question)
                            new_answer = generate_new(base_answer)
                            fout.write(json.dumps({'asin': asin, 'question': new_question, 'answer': new_answer, 'knowledge': knowledge, 'feature_name': feature_name, 'feature_value': feature_value, 'type': type}, ensure_ascii=False) + '\n')
                            fout.flush()

        with open(os.path.join(self.dirname, 'metadata_table.json'), 'w', encoding='utf-8') as fout_table:
            json.dump(metadata_table, fout_table, ensure_ascii=False, indent=4)

    def generate_short_answer(self):
        # generate short answer
        with open(os.path.join(self.dirname, 'product_qa_full.jsonl'), 'w', encoding='utf-8') as fout, open(os.path.join(self.dirname, 'product_reasoning_qa.jsonl'), 'r', encoding='utf-8') as fin:
            idx = 0
            for line in fin:
                print(f'Generating short answer for line {idx + 1} ...')
                data = json.loads(line)
                prompt = load_prompt(os.path.join('..', 'prompt', 'generate_short_yes_no_answer')).format(question=data['question'], answer=data['answer'])
                rst = self.client(prompt, model='gpt-4-1106-preview', max_tokens=10)
                short = 'yes'
                if 'no' in rst.lower():
                    short = 'no'
                full_data = {
                    'id': f'reasoning_qa_{idx}',
                    'asin': data['asin'],
                    'question': data['question'],
                    'long_answer': data['answer'],
                    'short_answer': short,
                    'type': 'reasoning_qa',
                    'extra_data': {
                        'knowledge': data['knowledge'],
                        'feature_name': data['feature_name'],
                        'feature_value': data['feature_value'],
                        'type': data['type']
                    }
                }
                fout.write(json.dumps(full_data, ensure_ascii=False) + '\n')
                fout.flush()
                idx += 1
        os.remove(os.path.join(self.dirname, 'product_reasoning_qa.jsonl'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--group', type=str, required=True, help='group name')
    arg_parser.add_argument('--group_filename', type=str, required=True, help='group filename')
    args = arg_parser.parse_args()

    reasoning_qa_generator = ReasoningQAGenerator(args.group, args.group_filename)
    reasoning_qa_generator.generate_knowledge()
    reasoning_qa_generator.generate_reasoning_qa()
    reasoning_qa_generator.generate_product_reasoning_qa()
    reasoning_qa_generator.generate_short_answer()
