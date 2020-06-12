# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
from copy import deepcopy
import json
import logging
import os
import tarfile
import tempfile
import socket
import csv
import io
import random

import torch

from transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"
EMPATHETICD_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
EMOTION = 2
CONTEXT = 3
UTTERANCE = 5

logger = logging.getLogger(__file__)

def replace_comma(text):
    p = {'_comma_':' ,', '..':'.', '...':'.'}
    for k, v in p.items():
        text = text.replace(k,v)
    return text

def get_candidates(csv_file):
    reader = csv.reader(csv_file)
    i = 0
    candidates = []
    for row in reader:
        if i == 0:
            i += 1
            continue
        candidates.append(replace_comma(row[5]))
        i += 1
    return candidates

def read_from_csv(csv_file, num_candidates):
    data = []
    i = 0
    last_emotion = None
    last_context = None
    all_candidates = get_candidates(csv_file)
    csv_file.seek(0)
    reader = csv.reader(csv_file)
    
    for row in reader:
        if i == 0:
            i += 1
            continue
        
        if row[EMOTION] == last_emotion:
            candidates = random.choices(all_candidates, k = num_candidates)
            candidates.append(replace_comma(row[UTTERANCE]))
            conversation = {"history": deepcopy(history), "candidates": candidates}
            utterances.append(conversation)
            history.append(replace_comma(row[UTTERANCE]))            
        else:
            # append dictionary to data and reinitialize it
            if last_emotion != None:
                entry = {"emotion": last_emotion, "context": last_context, "utterances": utterances}
                data.append(entry)
            
            # create data for a new entry in dataset
            last_emotion = row[EMOTION]
            last_context = replace_comma(row[CONTEXT])
            utterances = []
            history = [replace_comma(row[UTTERANCE])]
            
    return data

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def get_dataset(tokenizer, dataset_path, dataset_cache):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset

def get_empd_dataset(tokenizer, dataset_path, dataset_cache, num_candidates=1):
    """ Get tokenized EMPHATETIC DIALOGUES dataset from ParlAI or cache."""
    dataset_path = dataset_path or EMPATHETICD_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    
    if dataset_cache and os.path.isfile(dataset_cache):
        # print("AOLEU")
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        empd_file = cached_path(dataset_path)
        dataset = {}    
        with tarfile.open(empd_file, "r:gz") as tar:
            for member in tar:
                if member.isreg():      # Is it a regular file?
                    csv_file = io.StringIO(tar.extractfile(member).read().decode('utf-8'))
                    
                    if member.name == "empatheticdialogues/train.csv":
                        dataset["train"] = read_from_csv(csv_file, num_candidates)
                    if member.name == "empatheticdialogues/valid.csv":
                        dataset["valid"] = read_from_csv(csv_file, num_candidates)
                    if member.name == "empatheticdialogues/test.csv":
                        dataset["test"] = read_from_csv(csv_file, num_candidates)
        
        # with open('emp_dataset.csv', 'w') as csv_file:  
        #     writer = csv.writer(csv_file)
        #     for key, value in dataset.items():
        #         for l in value:
        #             for new_key, new_value in l.items():
        #                 writer.writerow([new_key, new_value])

        # return
        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir
