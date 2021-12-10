import argparse
import json
import logging
import os
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM)
from transformers.trainer_utils import is_main_process
from datasets import load_metric,Dataset
from utils import DataTrainingArguments, ModelArguments, load_json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTModel, CPTForConditionalGeneration
from modeling_bart import BartForConditionalGeneration,BartForMultiTaskFinetune


class GenDataset(torch.utils.data.Dataset):
    def __init__(self, args, file, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.seq_length = args.max_source_length

        self.pad_id = tokenizer.encode('[PAD]')[1]
        self.sep_id = tokenizer.encode('[PAD]')[1]
        self.bos_id = tokenizer.encode('[CLS]')[1]
        self.eos_id = tokenizer.encode('[SEP]')[1]
        self.emotion2id = {'开心':0,'悲伤':1,'惊讶':2,'生气':3,'others':4}

        self.input_ids, self.attention_mask, self.labels, self.labels_cls = self.process(file)
        assert len(self.input_ids) == len(self.attention_mask)

    def process(self, file):
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
        labels_cls = []
        for item in tqdm(file['data']):
            for position, dialog in enumerate(item['content']):
                if position == 0:
                    continue
                assert len(dialog)< self.seq_length
                with tokenizer.as_target_tokenizer():
                    token_ids_target = self.tokenizer.encode(dialog)
                labels.append(token_ids_target[1:])
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

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # print({
        #     "input_ids":self.input_ids[idx], 
        #     "attention_mask": self.attention_mask[idx], 
        #     "labels":self.labels[idx],
        #     "labels_cls":self.labels_cls[idx],
        # })
        return {
            "input_ids":self.input_ids[idx], 
            "attention_mask": self.attention_mask[idx], 
            "labels":self.labels[idx],
            # "labels_cls":self.labels_cls[idx],
        }

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='/path/to/model',type=str)
parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr",default=2e-5,type=float)
parser.add_argument("--batch_size",default='50',type=str)
parser.add_argument("--epoch",default='50',type=str)
parser.add_argument("--data_dir",default="/path/to/dataset/",type=str)

args = parser.parse_args()
arg_dict=args.__dict__

logger = logging.getLogger(__name__)

dataset_name=arg_dict['dataset']
outdir_1='output'
if not os.path.exists(outdir_1):
    os.mkdir(outdir_1)

outdir=outdir_1+'/'+dataset_name
if not os.path.exists(outdir):
    os.mkdir(outdir)

seed=len(os.listdir(outdir))+1
outdir=outdir+'/'+str(args.epoch)+'/'+str(seed)
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
    '--seed',str(1000*seed),
    '--num_train_epochs',arg_dict['epoch'],
    '--save_strategy','no',
    '--evaluation_strategy','epoch',
    '--learning_rate',str(arg_dict['lr']),
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

tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)
# model=CPTForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
model = BartForMultiTaskFinetune.from_pretrained(model_args.model_name_or_path,
                                )
model.config.max_length=data_args.val_max_target_length


if training_args.do_train:
    train_dataset = GenDataset(args = data_args, file=datasets['train'],tokenizer=tokenizer)

if training_args.do_eval:
    eval_dataset = GenDataset(args = data_args, file=datasets['validation'],tokenizer=tokenizer)

if training_args.do_predict:
    test_dataset = GenDataset(args = data_args, file=datasets['test'],tokenizer=tokenizer)



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

class TestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        metrics['epoch']=state.epoch
        state.log_history.append(metrics)

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    callbacks=[TestCallback],
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

if trainer.is_world_process_zero():
    if training_args.predict_with_generate:
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        test_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True,
        )
        test_preds = [pred.strip() for pred in test_preds]
        output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
        with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
            writer.write("\n".join(test_preds))
