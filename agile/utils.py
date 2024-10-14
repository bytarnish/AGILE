import json
import hashlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models import AzureClient, EmbeddingModel

embed_model = EmbeddingModel()
call_search_azure = AzureClient("Your-Azure-Key")

def get_group_split(group):
    return 'test' if group in ['all_pans', 'camera_cases', 'leggings', 'motherboards', 'rifle_scopes', 'rollerball_pens'] else 'train'


def text2emb(text):
    # implement the text to embedding function here 
    # input: text str
    # output: text embedding
    results = embed_model([text])
    return results[0]


def search_past_similar(question, history_qa_embed, memory_embed, qa_thred=0.53, memory_thred=0.42):
    question_emb = text2emb(question)
    if len(history_qa_embed) == 0:
        qs_num = 0
    else:
        dis = F.cosine_similarity(torch.tensor(history_qa_embed), torch.tensor(question_emb).unsqueeze(0), dim=1).cpu().numpy().tolist()
        qs_num = len([a for a in dis if a>qa_thred])
    if len(memory_embed) == 0:
        ms_num = 0
    else:
        dis = F.cosine_similarity(torch.tensor(memory_embed), torch.tensor(question_emb).unsqueeze(0), dim=1).cpu().numpy().tolist()
        ms_num = len([a for a in dis if a>memory_thred])
    return qs_num, ms_num


def search_k_memory(memory, memory_emb, query, tokenizer=None, target_num=1e8, target_length=512):
    # input:
    #   memory: list of memory data
    #   memory_emb: list of memory embedding
    #   query: query text
    #   target_num: the number of memory to be selected
    #   target_length: the length of memory to be selected
    # output:
    #   ans: the selected memory data

    assert len(memory) == len(memory_emb)
    if len(memory_emb) > 0:
        dis = []
        query_emb = text2emb(query)
        dis = F.cosine_similarity(torch.tensor(memory_emb), torch.tensor(query_emb).unsqueeze(0), dim=1).cpu().numpy().tolist() # [len(mem)]
        idx = range(len(dis))
        dis = list(zip(idx,dis))
        dis = sorted(dis, key=lambda x: x[1], reverse=True)
        data_sort = [memory[x[0]] for x in dis]
        if tokenizer is None:
            data_sort_decode = [x for x in data_sort]
        else:
            data_sort_decode = [tokenizer.encode(x) for x in data_sort]
        current_len, current_num, ans = 0, 0, ''
        for i in range(len(dis)):
            if current_num >= target_num:
                break
            if current_len + len(data_sort_decode[i]) > target_length:
                continue
            current_len += len(data_sort_decode[i])
            ans += data_sort[i]
            current_num += 1
        return ans
    else:
        return ''


def get_schema_description(schema):
    # generate the schema description for the prompt using from schema file
    schema_des = ''
    for key in schema:
        schema_des = schema_des + key
        if 'unit' in schema[key]:
            schema_des = schema_des + '[' + schema[key]['unit'] + ']'
        if schema[key]['type'] == 'choice':
            schema_des = schema_des + '('
            for value in schema[key]['choices']:
                schema_des = schema_des + value + ','
            schema_des = schema_des[:-1] + ')'
        schema_des = schema_des + '|'
    return schema_des[:-1]


def gen_hash_id(data):
    if not isinstance(data, bytes):
        data = str(data).encode('utf-8')
    m = hashlib.md5(data)
    md5sum = m.hexdigest()
    hash_id = int(md5sum, 16) % (2 ** 63)
    return hash_id


def get_action(resp):
    # parse the action from the response
    seekadvice_idx = resp.lower().find("seekadvice")
    searchproduct_idx = resp.lower().find("searchproduct")
    predictanswer_idx = resp.lower().find("predictanswer")
    idx_list = [[seekadvice_idx, " [SeekAdvice]\n"], [searchproduct_idx, " [SearchProduct]\n"], [predictanswer_idx, " [PredictAnswer]\n"]]
    action = [x for x in idx_list if x[0] >= 0]
    if len(action) == 0: # 提取失败则默认predictanswer
        action = [[predictanswer_idx, " [PredictAnswer]\n"]]
    action.sort(key=lambda x: x[0])
    return action[0][1]


def parse_answer(text, token):
    # extract the long and short answer
    begin_idx = text.find(token)
    text = text[begin_idx + len(token):]
    while len(text) > 0 and not ((text[0] >= 'a' and text[0] <= 'z') or (text[0] >= 'A' and text[0] <= 'Z')):
        text = text[1:]
    end_idx = text.find('[')
    if end_idx != -1:
        text = text[: end_idx]
    text = ''.join(text.split('\n'))
    return text


def get_sql(text):
    # extract the sql
    token = 'SELECT'
    s = text.split('\n')
    for line in s:
        idx = line.find(token)
        if idx == -1:
            continue
        return line[idx:], True
    return '', False


def load_prompt(file):
    # load the prompt from file
    prompt = ''
    with open(file) as fin:
        for line in fin:
            prompt = prompt + line
    return prompt


def eval_search(response, short_answer):
    # eval the search result
    if type(short_answer) is not list:
        # wrong action, wrong result
        return False
    if len(short_answer) == 0:
        if len(response) == 0:
            return True
        else:
            return False
    if len(response) == 0 and len(short_answer) != 0:
        return False
    answer_asin = [x["asin"] for x in short_answer]
    for i in response:
        if i not in answer_asin:
            return False
    return True


def eval_predict(response, short_answer, all_products):
    # eval the predict result in short answer
    if type(short_answer) is list:
        # comparison_qa but prediction answer
        response = response.lower()
        answer_asin = [x["asin"].lower() for x in short_answer]
        has_gold, not_has_not_gold = False, True # correct iff the answer contains gold and does not include not gold
        for i in short_answer:
            if i["asin"].lower() in response or i["title"].lower() in response:
                has_gold = True
                break
        for i in all_products:
            if i["asin"] in answer_asin:
                continue
            if i["asin"].lower() in response or i["title"].lower() in response:
                not_has_not_gold = False
                break
        return has_gold and not_has_not_gold
    else:
        return short_answer.lower().strip() == str(response).lower().strip()


def eval_predict_long(question, response, long_answer, prompt_file, azure_key):
    # eval the predict result in long answer
    # Note that this function needs GPT-4
    prompt = load_prompt(prompt_file).format(question=question, reference=long_answer, response=str(response))
    result = call_search_azure(prompt, "gpt-4", 500)
    if 'Yes' in result:
        return True, result
    return False, result


def str2dict(s):
    try:
        d = json.loads(s)
        return d
    except:
        try:
            d = json.loads(s[8:-3])
            return d
        except:
            pass

    begin_idx = s.find('```')
    if begin_idx < 0:
        print(s)
        return dict()

    end_idx = s[begin_idx + 3:].find('```')
    if end_idx < 0:
        print(s)
        return dict()

    while s[begin_idx] != '{':
        begin_idx += 1

    end_idx += 3
    while s[end_idx] != '}':
        end_idx -= 1

    try:
        d = json.loads(''.join(s[begin_idx: end_idx + 1].split('\n')))
        return d
    except:
        pass
    
    print(s)
    return dict()


def load_jsonl_in_json(json_file_path, encoding='utf-8', **kwargs):
    data = []
    with open(json_file_path, "r", encoding=encoding) as f:
        for line in f:
            data.append(json.loads(line.strip(), **kwargs))
    return data


def dump_jsonline(json_file_path, data, encoding="utf-8"):
    with open(json_file_path, "wt", encoding=encoding) as fout:
        for ins in data:
            fout.write(f"{json.dumps(ins, ensure_ascii=False)}\n")
    fout.close()
    return 0
