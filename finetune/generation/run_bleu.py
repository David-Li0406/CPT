import os
from utils import load_json
from transformers import BertTokenizer
from bleu_metric import Metric
import json
from rouge import Rouge

#需要计算bleu的文本文件所在的目录
arch='output'
tokenizer=BertTokenizer.from_pretrained('model/bart-base-chinese')
dataset='cconv/30'
# test_set=load_json('finetune/generation/cconv/test.json')
test_set = json.load(open('finetune/generation/cconv/test.json',encoding='utf8'))

labels=[]
for data in test_set['data']:
    for position, dialog in enumerate(data['content']):
        if position == 0:
            continue
        ids = tokenizer.encode(dialog)
        labels.append(tokenizer.decode(ids,skip_special_tokens=True))
labels=[[label.strip().split(' ')] for label in labels]
#     ids=tokenizer.encode(data['summarization'])
#     labels.append(tokenizer.decode(ids,skip_special_tokens=True))
# labels=[[label.strip().split(' ')] for label in labels]

def output(i, scores):
    print('bleu'+str(i),sum(scores)/len(scores))

rouge = Rouge()
metric=Metric(None)
idxs=os.listdir(os.path.join(arch,dataset))
bleu= [[],[],[],[],[]]
for idx in sorted(idxs):
    path=os.path.join(arch,dataset,idx,'test_generations.txt')
    with open(path,encoding='utf-8') as f:
        lines=f.readlines()
    lines=list(map(lambda x:x.strip(),lines))
    lines=[line.split(' ') for line in lines]
    metric.hyps=lines
    metric.refs=labels
    for i in range(1,5):
        bleu[i].append(metric.calc_bleu_k(i))
        output(i,bleu[i])
    labels = [l[0] for l in labels]
    labels = [' '.join(l) for l in labels]
    lines = [' '.join(l) for l in lines]
    rouge_score = rouge.get_scores(lines, labels,avg=True)
    print('rouge-1:',rouge_score["rouge-1"])
    print('rouge-2:',rouge_score["rouge-2"])
    print('rouge-l:',rouge_score["rouge-l"])