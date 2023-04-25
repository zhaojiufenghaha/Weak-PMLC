from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from src.model import LOTClassModel
from config.configs_interface import Configs
from src.data_utils.utils import load_stop_words
import warnings
import jieba, re
import itertools
from src.logers import LOGS

warnings.filterwarnings("ignore")


class WoBertTokenizer(BertTokenizer):
    def _tokenize(self, text):
        cut_words = jieba.lcut(text)
        return_words = []
        for w in cut_words:
            if w in self.vocab:
                # will not [UNK]
                return_words.append(w)
            else:
                # will be [UNK]
                w = list(w)
                return_words.extend(w)

        return return_words

    def tokenize(self, text, **kwargs):
        all_special_tokens = self.all_special_tokens

        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


class WeakPMLCTrainer(object):

    def __init__(self, args: Configs):
        self.args = args
        self.max_len = args.train_args.MAX_LEN
        self.dataset_dir = args.data.DATASET
        self.dist_port = args.train_args.dist_port
        self.num_cpus = min(4, cpu_count() - 4) if cpu_count() > 1 else 1
        self.world_size = args.train_args.GPUS
        self.train_batch_size = args.train_args.TRAIN_BATCH
        self.eval_batch_size = args.train_args.EVAL_BATCH
        self.accum_steps = args.train_args.ACCUM_STEP
        eff_batch_size = self.train_batch_size * self.world_size * self.accum_steps
        assert abs(
            eff_batch_size - 128) < 10, f"Make sure the effective training batch size is around 256, current: {eff_batch_size}"
        LOGS.log.debug(f"Effective training batch size: {eff_batch_size}")
        self.pretrained_lm = args.train_args.pretrained_weights_path
        jieba.load_userdict(os.path.join(self.pretrained_lm, 'vocab.txt'))  # load vocabulary
        self.tokenizer = WoBertTokenizer.from_pretrained(self.pretrained_lm)  # 分词
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        self.read_label_names(args.data.DATASET, args.data.LABEL_NAME_FILE)
        self.num_class = len(self.label_name_dict)
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=self.num_class)
        self.read_data(args.data.DATASET, args.data.TRAIN_CORPUS, args.data.TEST_CORPUS, args.data.TRAIN_LABEL,
                       args.data.TEST_LABEL)
        self.with_test_label = True if args.data.TEST_LABEL is not None else False
        self.temp_dir = f'tmp_{self.dist_port}'
        self.mcp_loss = nn.CrossEntropyLoss()
        self.st_loss = nn.KLDivLoss(reduction='batchmean')
        self.update_interval = args.train_args.update_interval
        self.early_stop = args.train_args.early_stop

    # set up distributed training
    def set_up_dist(self, rank):
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{self.dist_port}',
            world_size=self.world_size,
            rank=rank
        )
        # create local model
        model = self.model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        return model

    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True,
                                                        max_length=self.max_len,
                                                        padding='max_length',
                                                        pad_to_max_length=True,
                                                        return_attention_mask=True,
                                                        truncation=True,
                                                        return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # convert dataset into tensors
    # 将数据集转换为tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, find_label_name=False,
                       label_name_loader_name=None):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            LOGS.log.debug(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            LOGS.log.debug(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            # print('results', results)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            LOGS.log.debug(f"Saving encoded texts into {loader_file}")
            data = {"input_ids": input_ids, "attention_masks": attention_masks}
            torch.save(data, loader_file)
        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                LOGS.log.debug(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            else:
                LOGS.log.debug(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = [doc.strip() for doc in corpus.readlines()]
                LOGS.log.debug("Locating label names in the corpus.")
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x + chunk_size] for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(
                    delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
                input_ids_with_label_name = torch.cat([result[0] for result in results])
                attention_masks_with_label_name = torch.cat([result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
                label_name_data = {"input_ids": input_ids_with_label_name,
                                   "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                LOGS.log.debug(f"Saving texts with label names into {loader_file}")
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data

    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)  # 分词
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)  # 50个-1组成的tensor
        new_doc = []
        wordpcs = []
        idx = 1  # index starts at 1 due to [CLS] token  由于[CLS]令牌，索引从1开始
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1:  # last index will be [SEP] token  最后一个索引将是[SEP]令牌
                break
            if i == len(doc) - 1 or not doc[i + 1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label,
                                                            add_special_tokens=True,
                                                            max_length=self.max_len,
                                                            pad_to_max_length=True,
                                                            padding='max_length',
                                                            return_attention_mask=True,
                                                            truncation=True,
                                                            return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, test_file, train_label_file, test_label_file):
        self.train_data, self.label_name_data = self.create_dataset(dataset_dir, train_file, train_label_file, "train.pt",
                                                                    find_label_name=True,
                                                                    label_name_loader_name="label_name_data.pt")
        if test_file is not None:
            self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt")

    # read label names from file
    # 从文件中读取标签名称
    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower().strip() for word in category_words.strip().split()] for
                                i, category_words in
                                enumerate(label_names)}
        LOGS.log.debug(f"Label names used for each class are: {self.label_name_dict}")
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                # assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx  # {标签：序号}
                if word in self.vocab:  # 可以在bert词表里找到
                    self.all_label_name_ids.append(self.vocab[word])  # 标签存词在Bert词表中的的序号
                    self.all_label_names.append(word)  # 标签在Bert词表中出现的词

    # create dataset loader！！！！！！
    def make_dataloader(self, rank, data_dict, batch_size):
        if "labels" in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
        else:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)  # 可以约束数据只加载数据集的子集
        dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        return dataset_loader

    # 过滤掉停止词和多个类别中重复的词*****
    def filter_keywords(self, category_vocab_size=256):
        all_words = defaultdict(list)
        sorted_dicts = {}
        for i, cat_dict in self.category_words_freq.items():
            sorted_dict = {k: v for k, v in
                           sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)
        # print('repeat_words', repeat_words)
        repeat_words = []
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.category_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.category_vocab[i] = np.array(list(sorted_dict.keys()))  # np.array返回没有,的列表 返回sorted_dict.keys，即词表的index
        stopwords_vocab = load_stop_words(os.path.join(self.args.data.DATASET, self.args.data.stop_words))
        for i, word_list in self.category_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
                if word in self.label_name_dict[i]:
                    continue
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                    # isalpha用于判断一个字符串中，是否所有的字符都是字母，如果是返回True否则返回False
                    delete_idx.append(j)
            self.category_vocab[i] = np.delete(self.category_vocab[i], delete_idx)

    # construct category vocabulary (distributed function)
    # 构建类别词汇表（分布式函数）
    def category_vocabulary_dist(self, rank, top_pred_num=50, loader_name="category_vocab.pt"):
        model = self.set_up_dist(rank)
        model.eval()
        label_name_dataset_loader = self.make_dataloader(rank, self.label_name_data, self.eval_batch_size)
        category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
        wrap_label_name_dataset_loader = tqdm(label_name_dataset_loader) if rank == 0 else label_name_dataset_loader
        try:
            for batch in wrap_label_name_dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    label_pos = batch[2].to(rank)
                    match_idx = label_pos >= 0
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None,
                                        attention_mask=input_mask)
                    _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)  # torch.topk()返回列表中最大的50个值
                    label_idx = label_pos[match_idx]
                    for i, word_list in enumerate(sorted_res):
                        for j, word_id in enumerate(word_list):
                            category_words_freq[label_idx[i].item()][word_id.item()] += 1
                            # 将每个词的频率+1（字典嵌套）（初始化为1）
                            # .item()作用：取出单元素张量的元素值并返回该值，保持原元素类型不变。,即：原张量元素为整形，则返回整形，原张量元素为浮点型则返回浮点型
            save_file = os.path.join(self.temp_dir, f"{rank}_" + loader_name)
            torch.save(category_words_freq, save_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # construct category vocabulary
    # 构建类别词汇表
    def category_vocabulary(self, top_pred_num=50, category_vocab_size=100, loader_name="category_vocab.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            LOGS.log.debug(f"Loading category vocabulary from {loader_file}")
            self.category_vocab = torch.load(loader_file)
        else:
            LOGS.log.debug("Contructing category vocabulary.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.category_vocabulary_dist, nprocs=self.world_size, args=(top_pred_num, loader_name))  # 分布式训练
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            self.category_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
            for i in range(self.num_class):
                for category_words_freq in gather_res:
                    for word_id, freq in category_words_freq[i].items():
                        self.category_words_freq[i][word_id] += freq
                        # 出现就+1
            self.filter_keywords(category_vocab_size)  # 过滤掉停止词和多个类别中重复的词
            torch.save(self.category_vocab, loader_file)
            with open(loader_file.replace('.pt', '.txt'), mode='w', encoding='utf-8') as wf:
                for i, wk in self.category_vocab.items():
                    wk = wk.tolist()
                    wk = [str(self.inv_vocab[w]) for w in wk]
                    wl = ' '.join(wk)
                    wf.write(str(self.label_name_dict[i][0])+'=='+wl + '\n')
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)  # 递归地删除文件
        for i, category_vocab in self.category_vocab.items():
            LOGS.log.debug(
                f"Class {self.label_name_dict[i]} category vocabulary: {[self.inv_vocab[w] for w in category_vocab]}\n")

    def cuda_mem_error(self, err, mode, rank):
        if rank == 0:
            LOGS.log.debug(err)
            if "CUDA out of memory" in str(err):
                if mode == "eval":
                    LOGS.log.debug(
                        f"Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size`, current: {self.eval_batch_size}")
                else:
                    LOGS.log.debug(
                        f"Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size`, current: {self.train_batch_size}")
        sys.exit(1)
