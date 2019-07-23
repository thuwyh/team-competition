import argparse
from collections import defaultdict, Counter
import random

import pandas as pd
import tqdm
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = Path('../byebyejuly')

def make_folds(n_folds: int) -> pd.DataFrame:
    df = pd.read_table(DATA_ROOT / 'train.txt', names=['a', 'b', 'label'])

    kf = StratifiedKFold(n_splits=n_folds, random_state=2019)
    df['fold'] = -1
    for folds, (train_index, test_index) in enumerate(kf.split(df['a'], df['label'])):
        df.loc[test_index, 'fold'] = folds

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_pickle(DATA_ROOT/'folds.pkl')


if __name__ == '__main__':
    main()
