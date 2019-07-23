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
from bert_dataset import TrainDataset, collate_fn
from model import PairModel
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam
from sklearn.metrics import accuracy_score, roc_auc_score

BERT_PRETRAIN_PATH = '../byebyejuly/bert-chinese-wwm/'

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict'])
    arg('run_root')
    arg('--batch-size', type=int, default=64)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2)
    arg('--lr', type=float, default=0.00002)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=5)
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)
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

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=0.05,
                             t_total=len(training_loader) // args.step)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", verbosity=0)
        optimizer.zero_grad()

        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        train(args, model, optimizer, None,
              train_loader=training_loader,
              valid_df = valid_fold,
              valid_loader=valid_loader, epoch_length=len(training_set))

    elif args.mode == 'validate':
        pass
        # valid_set = TrainDataset(valid_fold['comment_text'].tolist(), lens=valid_fold['len'].tolist(),
        #                          weights=valid_fold['weights'].tolist(), target=valid_fold['binary_target'].tolist())
        # valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        #                           num_workers=args.workers)
        # model = BertModel(BERT_PRETRAIN_PATH)
        # load_model(model, run_root / ('best-model-%d.pt' % args.fold), multi2single=False)
        # model.cuda()
        # if args.multi_gpu == 1:
        #     model = nn.DataParallel(model)
        # validation(model, valid_df, valid_loader, args, False, progress=True)


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
        for i, (input_a, input_b, targets) in enumerate(train_loader):
            mask_a = (input_a > 0).cuda()
            mask_b = (input_b > 0).cuda()
            input_a, input_b, targets = input_a.cuda(), input_b.cuda(), targets.cuda()

            outputs = model(input_a, mask_a, input_b, mask_b)
            loss = loss_fn(outputs, targets.view(-1,1)) / args.step
            batch_size = input_a.size(0)

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
        for input_a, input_b, targets in valid_loader:
            if progress:
                batch_size = input_a.size(0)
                tq.update(batch_size)
            all_targets.append(targets.numpy().copy())

            mask_a = (input_a > 0).cuda()
            mask_b = (input_b > 0).cuda()
            input_a, input_b, targets = input_a.cuda(), input_b.cuda(), targets.cuda()

            outputs = model(input_a, mask_a, input_b, mask_b)

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
    metrics['acc'] = accuracy_score(all_targets, all_predictions>0.5)
    metrics['auc'] = roc_auc_score(all_targets, all_predictions)
    to_print = []
    for idx, (k, v) in enumerate(sorted(metrics.items(), key=lambda kv: -kv[1])):
        to_print.append(f'{k} {v:.5f}')
    print(' | '.join(to_print))
    return metrics


if __name__ == '__main__':
    main()