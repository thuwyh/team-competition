import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm

from torch import nn
from torch.utils.data import DataLoader
from apex import amp

from utils import (
    BucketBatchSampler,
    get_learning_rate, set_learning_rate, set_seed,
    write_event, load_model)
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset
import random
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict

BERT_PRETRAIN_PATH = '../byebyejuly/bert-chinese-wwm/'
VOCAB_PATH = Path('../byebyejuly/bert-chinese-wwm/vocab.txt')


def convert_one_line(text_a, text_b, tokenizer=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b)
    token_type = np.zeros(len(one_token))
    token_type[-len(tokens_b):] = 1
    return one_token, token_type


class TrainDataset(Dataset):

    def __init__(self, data, vocab_path=VOCAB_PATH, do_lower=True, shuffle=False):
        super(TrainDataset, self).__init__()
        self._a = data['a'].tolist()
        self._b = data['b'].tolist()
        self._label = data['label'].tolist()
        self._shuffle = shuffle
        self._tokenizer = BertTokenizer.from_pretrained(
            vocab_path, cache_dir=None, do_lower_case=do_lower)

    def __len__(self):
        return len(self._a)

    def __getitem__(self, idx):
        if self._shuffle and random.random() < 0.5:
            token, token_type = convert_one_line(self._b[idx], self._a[idx], tokenizer=self._tokenizer)
        else:
            token, token_type = convert_one_line(self._a[idx], self._b[idx], tokenizer=self._tokenizer)
        return torch.LongTensor(token), torch.LongTensor(token_type), self._label[idx]


def collate_fn(batch):
    token, token_type, targets = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    targets = torch.FloatTensor(targets)
    return token, token_type, targets


class PairModel(nn.Module):

    def __init__(self, pretrain_path, dropout=0.1):
        super(PairModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrain_path, cache_dir=None, num_labels=1)
        self.head = nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(dropout)),
                ('clf', nn.Linear(self.bert.config.hidden_size, 1)),
            ])
        )

    def forward(self, inputs, token_type, masks, token_type_ids=None):
        _, pooled_output = self.bert.bert(inputs, token_type, masks, output_all_encoded_layers=False)
        out = self.head(pooled_output)
        return out


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict'])
    arg('run_root')
    arg('--batch-size', type=int, default=64)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2)
    arg('--lr', type=float, default=0.00001)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=5)
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)
    arg('--lr_layerdecay', type=float, default=0.95)
    args = parser.parse_args()

    set_seed()

    run_root = Path('../experiments/' + args.run_root)
    DATA_ROOT = Path('../byebyejuly/')

    folds = pd.read_pickle(DATA_ROOT / 'folds.pkl')
    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        training_set = TrainDataset(train_fold, do_lower=True, shuffle=True)

        training_loader = DataLoader(training_set, collate_fn=collate_fn, shuffle=True, batch_size=args.batch_size,
                                     num_workers=args.workers)

        valid_set = TrainDataset(valid_fold)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  num_workers=args.workers)

        model = PairModel(BERT_PRETRAIN_PATH)
        model.cuda()

        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
        #      'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
        #      'weight_decay': 0.0}
        # ]
        NUM_LAYERS = 12
        optimizer_grouped_parameters = [
            {'params': model.bert.bert.embeddings.parameters(), 'lr': args.lr * (args.lr_layerdecay ** NUM_LAYERS)},
            {'params': model.head.parameters(), 'lr': args.lr},
            {'params': model.bert.bert.pooler.parameters(), 'lr': args.lr}
        ]

        for layer in range(NUM_LAYERS):
            optimizer_grouped_parameters.append(
                {'params': model.bert.bert.encoder.layer.__getattr__('%d' % (NUM_LAYERS - 1 - layer)).parameters(),
                 'lr': args.lr * (args.lr_layerdecay ** layer)},
            )
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=0.05,
                             t_total=len(training_loader) * args.n_epochs // args.step)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", verbosity=0)
        optimizer.zero_grad()

        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        train(args, model, optimizer, None,
              train_loader=training_loader,
              valid_df=valid_fold,
              valid_loader=valid_loader, epoch_length=len(training_set))

    elif args.mode == 'validate':
        valid_fold = pd.read_table('../byebyejuly/test.txt', names=['a', 'b', 'label'])

        valid_set = TrainDataset(valid_fold)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  num_workers=args.workers)

        model = PairModel(BERT_PRETRAIN_PATH)
        load_model(model, run_root / ('best-model-%d.pt' % args.fold), multi2single=False)
        model.cuda()
        if args.multi_gpu == 1:
            model = nn.DataParallel(model)
        validation(model, valid_fold, valid_loader, args, False, progress=True)


def train(args, model: nn.Module, optimizer, scheduler, *,
          train_loader, valid_df, valid_loader, epoch_length, n_epochs=None) -> bool:
    n_epochs = n_epochs or args.n_epochs

    run_root = Path('../experiments/' + args.run_root)
    model_path = run_root / ('model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    if best_model_path.exists():
        state, best_valid_score = load_model(model, best_model_path)
        start_epoch = state['epoch']
        best_epoch = start_epoch
    else:
        best_valid_score = 0
        start_epoch = 0
        best_epoch = 0
    step = 0

    save = lambda ep: torch.save({
        'model': model.module.state_dict() if args.multi_gpu == 1 else model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': current_score
    }, str(model_path))
    #
    report_each = 10000
    log = run_root.joinpath('train-%d.log' % args.fold).open('at', encoding='utf8')

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()

        lr = get_learning_rate(optimizer)
        tq = tqdm.tqdm(total=epoch_length)
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []

        mean_loss = 0
        loss_fn = nn.BCEWithLogitsLoss()
        for i, (inputs, token_types, targets) in enumerate(train_loader):
            masks = (inputs > 0).cuda()

            inputs, token_types, targets = inputs.cuda(), token_types.cuda(), targets.cuda()

            outputs = model(inputs, token_types, masks)
            loss = loss_fn(outputs, targets.view(-1, 1)) / args.step
            batch_size = inputs.size(0)

            # loss.backward()
            if (i + 1) % args.step == 0:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                with amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

            tq.update(batch_size)
            losses.append(loss.item() * args.step)
            mean_loss = np.mean(losses[-report_each:])
            tq.set_postfix(loss=f'{mean_loss:.5f}')
            if i and i % report_each == 0:
                write_event(log, step, loss=mean_loss)

        write_event(log, step, epoch=epoch, loss=mean_loss)
        tq.close()

        # if epoch<7: continue
        valid_metrics = validation(model, valid_df, valid_loader, args)
        write_event(log, step, **valid_metrics)
        current_score = valid_metrics['auc']
        save(epoch + 1)
        if scheduler is not None:
            scheduler.step(current_score)
        if current_score > best_valid_score:
            best_valid_score = current_score
            shutil.copy(str(model_path), str(best_model_path))
            best_epoch = epoch
        else:
            pass
    return True


def validation(model: nn.Module, valid_df, valid_loader, args, save_result=False, progress=False) -> Dict[
    str, float]:
    run_root = Path('../experiments/' + args.run_root)
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for inputs, token_types, targets in valid_loader:
            if progress:
                batch_size = inputs.size(0)
                tq.update(batch_size)
            all_targets.append(targets.numpy().copy())

            masks = (inputs > 0).cuda()

            inputs, token_types, targets = inputs.cuda(), token_types.cuda(), targets.cuda()

            outputs = model(inputs, token_types, masks)

            predictions = torch.sigmoid(outputs[:, 0].view(-1, 1))
            all_predictions.append(predictions.cpu().numpy())

            outputs = outputs[:, 0].view(-1, 1)
            loss = nn.BCEWithLogitsLoss()(outputs,
                                          targets.view(-1, 1))  # criterion(outputs, targets).mean()  # *N_CLASSES
            all_losses.append(loss.item())  # _reduce_loss

    all_predictions = np.concatenate(all_predictions)

    all_targets = np.concatenate(all_targets)

    if save_result:
        np.save(run_root / 'prediction_fold{}.npy'.format(args.fold), all_predictions)
        np.save(run_root / 'target_fold{}.npy'.format(args.fold), all_targets)

    metrics = dict()
    metrics['loss'] = np.mean(all_losses)
    metrics['acc'] = accuracy_score(all_targets, all_predictions > 0.5)
    metrics['auc'] = roc_auc_score(all_targets, all_predictions)
    to_print = []
    for idx, (k, v) in enumerate(sorted(metrics.items(), key=lambda kv: -kv[1])):
        to_print.append(f'{k} {v:.5f}')
    print(' | '.join(to_print))
    return metrics


if __name__ == '__main__':
    main()
