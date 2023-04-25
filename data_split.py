import pandas as pd
from tqdm import tqdm
import random


def train_test_split():
    with open('Data.txt', 'r', encoding='utf-8') as rf:
        Datas = [each.strip() for each in rf.readlines()]
    random.seed(9)
    random.shuffle(Datas)
    split = int(len(Datas) * 0.64)
    Trains = Datas[:split]
    Tests = Datas[split:]

    with open('train.txt', 'w', encoding='utf-8') as wf, open('test.txt', 'w', encoding='utf-8') as wf:
        for train in tqdm(Trains):
            train = train.split('\t')
            wf.write(train[0] + '\t' + train[1])

        for test in tqdm(Tests):
            test = test.split('\t')
            wf.write(test[0] + '\t' + test[1])


train_test_split()