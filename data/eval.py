import argparse
import json
import openai
import os
import time

long_eval_prompt = """
Based on the provided question and reference answer, please determine if the response is correct or incorrect. Begin by articulating your rationale, and conclude with a single word judgment: 'Yes' for correct or 'No' for incorrect.

question: {question}
reference answer: {reference}
response: {response}
"""


def long_eval(gold, pred, question):
    if not isinstance(pred, str):
        return False
    while 1:
        try:
            response = openai.chat.completions.create(
                model='gpt-4',
                messages=[
                    {
                        'role': 'user',
                        'content': long_eval_prompt.format(question=question, reference=gold, response=pred).strip()
                    }
                ],
                max_tokens=300,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return 'yes' in response['choices'][0]['message']['content'].lower()
        except Exception as e:
            print(e)
            time.sleep(10)


def short_eval(gold, pred):
    if not isinstance(pred, str):
        return False
    return ' '.join(gold.lower().split()) == ' '.join(pred.lower().split())


def search_eval(gold, pred):
    if not isinstance(pred, list):
        return False
    if len(gold) == 0:
        return len(pred) == 0
    if len(pred) == 0:
        return False
    for item in pred:
        if item not in gold:
            return False
    return True


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_dir', type=str, required=True, help='The directory of the predictions from your model.')
    arg_parser.add_argument('--output_dir', type=str, required=True, help='The directory of the evaluation results.')
    arg_parser.add_argument('--long_eval', action='store_true', help='Whether evaluating long answers or not.')
    args = arg_parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result_logger = open(os.path.join(args.output_dir, 'result.txt'), 'w', encoding='utf-8')
    result_logger.write('Group'.ljust(15) + 'Long Eval'.rjust(13) + 'Short Eval'.rjust(14) + '\n')
    total_num_cnt, total_long_cnt, total_short_cnt = 0, 0, 0
    for group in ['all_pans', 'camera_cases', 'leggings', 'motherboards', 'rifle_scopes', 'rollerball_pens']:
        if not os.path.exists(os.path.join(args.input_dir, group + '.jsonl')):
            print(f'Prediction file for {group} not found.')
            continue
        golds, preds = [], []
        with open(os.path.join('test', group, 'qa.jsonl'), 'r', encoding='utf-8') as file:
            for line in file:
                golds.append(json.loads(line))
        with open(os.path.join(args.input_dir, group + '.jsonl'), 'r', encoding='utf-8') as file:
            for line in file:
                preds.append(json.loads(line))
        cur_long_cnt, cur_short_cnt = 0, 0
        with open(os.path.join(args.output_dir, group + '.jsonl'), 'w', encoding='utf-8') as file:
            for gold, pred in zip(golds, preds):
                if gold['type'] == 'search_qa':
                    pred['short_eval'] = search_eval([x['asin'] for x in gold['short_answer']], pred['short_answer'])
                    if args.long_eval:
                        pred['long_eval'] = pred['short_eval']
                else:
                    pred['short_eval'] = short_eval(gold['short_answer'], pred['short_answer'])
                    if args.long_eval:
                        pred['long_eval'] = long_eval(gold['long_answer'], pred['long_answer'], gold['question'])
                cur_short_cnt += int(pred['short_eval'])
                if args.long_eval:
                    cur_long_cnt += int(pred['long_eval'])
                file.write(json.dumps(pred, ensure_ascii=False) + '\n')
        total_num_cnt += len(golds)
        total_long_cnt += cur_long_cnt
        total_short_cnt += cur_short_cnt
        result_logger.write(group.ljust(15) + (str(round(cur_long_cnt / len(golds), 3)) if args.long_eval else '-').rjust(13) + str(round(cur_short_cnt / len(golds), 3)).rjust(14) + '\n')
    result_logger.write('-' * 42 + '\n')
    result_logger.write('Total'.ljust(15) + (str(round(total_long_cnt / total_num_cnt, 3)) if args.long_eval else '-').rjust(13) + str(round(total_short_cnt / total_num_cnt, 3)).rjust(14))
    result_logger.close()
