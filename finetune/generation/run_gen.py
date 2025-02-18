import argparse
import json
import logging
import os
import random
import sys
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer, TrainingArguments, 
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM)
from transformers.trainer_utils import is_main_process
from datasets import load_metric,Dataset
from utils import DataTrainingArguments, ModelArguments, load_json
from transformers import EarlyStoppingCallback

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTModel, CPTForConditionalGeneration
from modeling_bart import BartForConditionalGeneration,BartForMultiTaskFinetune


class GenDataset(torch.utils.data.Dataset):
    def __init__(self, args, file, tokenizer: BertTokenizer, cls_emo, cls_mode, add_enc=True, mode='turn', turn_num=6, prompt=True):
        if add_enc and cls_emo:
             raise ValueError("Can't add emotion labels to encoder and predict it in the same time")
        self.tokenizer = tokenizer
        if mode == 'length':
            self.seq_length = args.max_source_length
        else:
            self.turn_num = turn_num
        self.cls_emo = cls_emo
        self.cls_mode = cls_mode
        self.add_enc = add_enc
        self.prompt = prompt

        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.encode('[CLS]')[1]
        self.eos_id = self.tokenizer.encode('[SEP]')[1]
        self.mask_id = self.tokenizer.mask_token_id

        self.emotion2id = {'开心':0,'悲伤':1,'惊讶':2,'生气':3,'others':4}


        self.token2id = {
            "bot": self.tokenizer.encode('[bot]')[1],
            "eot":self.tokenizer.encode('[eot]')[1],
            "开心":self.tokenizer.encode('[开心]')[1],
            "悲伤":self.tokenizer.encode('[悲伤]')[1],
            "惊讶":self.tokenizer.encode('[惊讶]')[1],
            "生气":self.tokenizer.encode('[生气]')[1],
            "others":self.tokenizer.encode('[others]')[1],
            "noLabel":self.tokenizer.encode('[noLabel]')[1],
        }

        if mode == 'turn':
            self.input_ids, self.attention_mask, self.labels, self.labels_cls = self.process_with_turn(file)
        else:
            self.input_ids, self.attention_mask, self.labels, self.labels_cls = self.process_with_length(file)
        assert len(self.input_ids) == len(self.attention_mask)

    def process_with_turn(self, file):
        input_ids = []
        attention_mask = []
        labels  = []
        labels_cls = None
        if self.cls_emo:
            labels_cls = []
        for item in tqdm(file['data'][:2]):
            for position, dialog in enumerate(item['content']):
                if position == 0:
                    continue
                with self.tokenizer.as_target_tokenizer():
                    token_ids_target = self.tokenizer.encode(dialog)
                labels.append(token_ids_target[1:])
                if self.cls_emo:
                    if position-1 >= 0 and item['emotion'][position-1] != 0:
                        labels_cls.append([self.emotion2id[item['emotion'][position-1]]])
                    else:
                        labels_cls.append([-1])
                _position = position
                cur_turn = 0
                token_ids_input = []
                while _position > 0 and cur_turn < self.turn_num:
                    if self.add_enc:
                        emotion = item['emotion'][_position]
                        if emotion == 0:
                            emotion = 'noLabel'
                        token_ids_input = [self.token2id['bot']] + self.tokenizer.encode(item['content'][_position-1])[1:-1] + [self.token2id[emotion], self.token2id['eot']] + token_ids_input
                    else:
                        token_ids_input = [self.token2id['bot']] + self.tokenizer.encode(item['content'][_position-1])[1:-1] + [self.token2id['eot']] + token_ids_input
                    _position -= 1
                    cur_turn += 1
                # print(token_ids_input)
                input_id = [self.bos_id] + token_ids_input + [self.eos_id]
                if self.prompt:
                    input_id = self.add_template(input_id)
                input_ids.append(input_id)
                attention_mask.append([1 for _ in input_id])
                # print(len(input_ids), len(attention_mask))
                assert len(input_ids[-1]) == len(attention_mask[-1])
        return input_ids, attention_mask, labels, labels_cls

    def process_with_length(self, file):
        '''
        data:{
            'data':[
                {
                    'content':[,],
                    'class':,
                    'scenario:,'
                }
            ]
        }
        '''
        input_ids = []
        attention_mask = []
        labels  = []
        labels_cls = None
        if self.cls_emo:
            labels_cls = []
        for item in tqdm(file['data']):
            for position, dialog in enumerate(item['content']):
                if position == 0:
                    continue
                assert len(dialog)< self.seq_length
                with self.tokenizer.as_target_tokenizer():
                    token_ids_target = self.tokenizer.encode(dialog)
                labels.append(token_ids_target[1:])
                if self.cls_emo:
                    if position-1 >= 0 and item['emotion'][position-1] != 0:
                        labels_cls.append([self.emotion2id[item['emotion'][position-1]]])
                    else:
                        labels_cls.append([-1])
                _position = position
                cur_length = 0
                token_ids_input = []
                while _position > 0 and cur_length + len(self.tokenizer.encode(item['content'][_position-1])) + 3 < self.seq_length: 
                    token_ids_input = self.tokenizer.encode(item['content'][_position-1])[1:-1] + [self.sep_id] + token_ids_input
                    cur_length = len(token_ids_input)
                    _position -= 1
                input_ids.append([self.bos_id] + token_ids_input[:-1] + [self.eos_id] + [self.pad_id] * (self.seq_length - len(token_ids_input)))
                attention_mask.append([1 for _ in range(len(token_ids_input)+1)]+[0 for _ in range(self.seq_length - len(token_ids_input))])
                assert len(input_ids[-1]) == len(attention_mask[-1])
        return input_ids, attention_mask, labels, labels_cls

    def add_template(self, input_id, method="discrete", template="对话人感到[MASK]"):
        template_id = self.tokenizer.encode(template)[1:-1]
        input_id = input_id[:-1] + template_id + [input_id[-1]]
        return input_id


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # print(self.input_ids[idx])
        if self.cls_emo:
            return {
                "input_ids":self.input_ids[idx], 
                "attention_mask": self.attention_mask[idx], 
                "labels":self.labels[idx],
                "labels_cls":self.labels_cls[idx],
            }
        else:
            return{
                "input_ids":self.input_ids[idx], 
                "attention_mask": self.attention_mask[idx], 
                "labels":self.labels[idx],  
            }

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='/path/to/model',type=str)
parser.add_argument("--model_name",default='bart-base-chinese-finetune',type=str)
parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr",default=2e-5,type=float)
parser.add_argument("--batch_size",default='50',type=str)
parser.add_argument("--epoch",default='50',type=str)
parser.add_argument("--data_dir",default="/path/to/dataset/",type=str)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--cls_emo", default=False, type=bool)
parser.add_argument("--add_enc", default=False, type=bool)
parser.add_argument("--cls_mode", default=1, type=int)
parser.add_argument("--gen_csk", default=False, type=bool)
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--beta", default=0.0, type=float)
parser.add_argument("--omega", default=0.0, type=float)

args = parser.parse_args()
arg_dict=args.__dict__
# local_rank = args.local_rank
# torch.cuda.set_device(local_rank)

logger = logging.getLogger(__name__)

dataset_name=arg_dict['dataset']
outdir_base='output'
if not os.path.exists(outdir_base):
    os.mkdir(outdir_base)

outdir=outdir_base+'/'+dataset_name
if not os.path.exists(outdir):
    os.mkdir(outdir)

seed=len(os.listdir(outdir))+1
outdir=outdir+'/'+args.model_name
length_map={'lcsts':'30','csl':'50','adgen':'128','cconv':'128'}


args=[
    '--model_name_or_path',arg_dict['model_path'],
    '--do_train','--do_eval','--do_predict',
    '--train_file',os.path.join(arg_dict['data_dir'],'train.json'),
    '--validation_file',os.path.join(arg_dict['data_dir'],'valid.json'),
    '--test_file',os.path.join(arg_dict['data_dir'],'test.json'),
    '--output_dir',outdir,
    '--per_device_train_batch_size',arg_dict['batch_size'],
    '--per_device_eval_batch_size',arg_dict['batch_size'],
    '--overwrite_output_dir',
    '--max_source_length=128',
    '--val_max_target_length='+length_map[arg_dict['dataset']],
    '--predict_with_generate=1',
    '--seed',str(args.local_rank+1),
    '--num_train_epochs',arg_dict['epoch'],
    '--save_strategy','no',
    '--evaluation_strategy','epoch',
    '--learning_rate',str(arg_dict['lr']),
    '--load_best_model_at_end',str(True), 
    '--metric_for_best_model',"eval_rouge-l", 
    '--greater_is_better',str(True),
    '--save_total_limit', str(1),
    '--cls_emo', str(args.cls_emo),
    '--add_enc', str(args.add_enc),
    '--cls_mode', str(args.cls_mode),
    '--gen_csk', str(args.gen_csk),
    '--beta',str(args.beta),
    '--alpha',str(args.alpha),
    '--omega',str(args.omega),
]

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_seed(training_args.seed)

datasets={}
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file
for key in data_files:
    print(key)
    print(data_files[key])
    datasets[key]=json.load(open(data_files[key],'r',encoding='utf8'))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/evaluation parameters %s", training_args)

def add_emotion_token(tokenizer, model):
    tokenizer.add_special_tokens({'additional_special_tokens':["<happy>","<sad>","<surprise>","<angry>","<others>"]})
    model.resize_token_embedding(len(tokenizer))

tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)
tokenizer.add_special_tokens({'additional_special_tokens':["[bot]","[eot]","[开心]","[悲伤]","[惊讶]","[生气]","[others]","[noLael]"]})
model = BartForMultiTaskFinetune.from_pretrained(model_args.model_name_or_path, 
                                                tokenizer = tokenizer, 
                                                cls_emo = model_args.cls_emo, 
                                                cls_mode = int(model_args.cls_mode),
                                                gen_csk = model_args.gen_csk,
                                                alpha = float(model_args.alpha),
                                                beta = float(model_args.beta),
                                                omega = float(model_args.omega),)
model.config.max_length=data_args.val_max_target_length

# device = torch.device("cuda", local_rank)
# model.to(device)
# model = DDP(model, device_ids=[local_rank], output_device=local_rank)


if training_args.do_train:
    # print(model_args.cls_emo)
    train_dataset = GenDataset(args = data_args, file=datasets['train'], tokenizer=tokenizer, cls_emo=model_args.cls_emo, cls_mode=model_args.cls_mode, add_enc=model_args.add_enc)

if training_args.do_eval:
    eval_dataset = GenDataset(args = data_args, file=datasets['validation'],tokenizer=tokenizer, cls_emo=model_args.cls_emo, cls_mode=model_args.cls_mode, add_enc=model_args.add_enc)

if training_args.do_predict:
    test_dataset = GenDataset(args = data_args, file=datasets['test'],tokenizer=tokenizer, cls_emo=model_args.cls_emo, cls_mode=model_args.cls_mode, add_enc=model_args.add_enc)

max_eval_num=30000
if len(eval_dataset)>max_eval_num:
    eval_dataset=Dataset.from_dict(eval_dataset[:max_eval_num])
print(len(eval_dataset))


# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)



# Metric
from rouge import Rouge 
rouge = Rouge()

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    while '' in preds:
        idx=preds.index('')
        preds[idx]='。'

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
   
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    scores = rouge.get_scores(decoded_preds, decoded_labels,avg=True)
    for key in scores:
        scores[key]=scores[key]['f']*100

    result=scores

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# class TestCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
#         metrics['epoch']=state.epoch
#         state.log_history.append(metrics)

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    callbacks=[EarlyStoppingCallback(10,0.0)],
)


# Training
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if training_args.predict_with_generate:
    predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
    if trainer.is_world_process_zero():
        test_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True,
        )
        test_preds = [pred.strip() for pred in test_preds]
        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
        with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
            writer.write("\n".join(test_preds))
