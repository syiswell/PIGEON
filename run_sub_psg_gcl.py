import os, time, random, sys, json
import numpy as np
import logging
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from modeling.ptgmodel_psg_gcl import PTGModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam, Adamax
import datetime
from configs.config import shared_configs
from dataloaders.dataloader_sent_psg_gcl import Instance, DataIterator, load_data_instances
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm, trange
from utils.misc import set_random_seed, NoOp, zero_none_grad
from utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
import math
from os.path import join
from utils.load_save import (ModelSaver, save_training_meta, load_state_dict_with_mismatch, E2E_TrainingRestorer)
from utils.basic_utils import load_json, get_index_positions, flat_list_of_lists
from utils.utils import context_models, args_metric, extract_arguments
from torch.nn.utils import clip_grad_norm_
import pickle


def setup_model(cfg):
    LOGGER.info('Initializing model...')
    model = PTGModel(cfg, "bert")

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    # else:
    # 	LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
    # 	LOGGER.info(f"Loading udg weights from {cfg.udg_weights_path}")
    # 	model.load_separate_ckpt(
    # 		bert_weights_path=cfg.bert_weights_path,
    # 		udg_weights_path=cfg.udg_weights_path
    # 	)

    # if cfg.freeze_bert:
    # 	model.freeze_bert_backbone()
    model.to(cfg.device)

    LOGGER.info('Model initialized.')

    for n, p in model.named_parameters():
        print(n, p.size())

    return model


def setup_dataloaders(config, tokenizer):
    LOGGER.info('Loading data...')

    train_sentence_packs = eval(json.load(open(config.data_path + '/train3.json')))
    dev_sentence_packs = eval(json.load(open(config.data_path + '/dev3.json')))
    test_sentence_packs = eval(json.load(open(config.data_path + '/test3.json')))

    if config.metric == 'poincare':
        threshold = config.poincare_threshold  # default: 0.7
    elif config.metric == 'dist':
        threshold = config.dist_threshold  # default: 0.2

    postfix = f"_graph_matrix_probing_mask_{config.probing_mask}_" \
                               f"metric_{str(config.metric)}_threshold_{threshold}_" \
                               f"max_cor_num_type_{str(config.max_cor_num_type)}_" \
                               f"inter_cor_threshold_{config.inter_cor_threshold}_" \
                               f"intra_neighbour_{config.intra_neighbour}_" \
                               f"intra_cor_threshold{config.intra_cor_threshold}{config.noone}.json"
    train_graph_matrices = json.load(open(config.data_path + '/ablation/train' + postfix))
    dev_graph_matrices = json.load(open(config.data_path + '/ablation/dev' + postfix))
    test_graph_matrices = json.load(open(config.data_path + '/ablation/test' + postfix))

    if cfg.encoding_scheme == 'IOBES':
        cfg.num_tags = 5
    else:
        cfg.num_tags = 3

    instances_train = load_data_instances(train_sentence_packs, train_graph_matrices, tokenizer, config)
    instances_dev = load_data_instances(dev_sentence_packs, dev_graph_matrices, tokenizer, config)
    instances_test = load_data_instances(test_sentence_packs, test_graph_matrices, tokenizer, config)

    del train_sentence_packs
    del dev_sentence_packs
    del test_sentence_packs

    random.shuffle(instances_train)

    LOGGER.info('Data loaded.')

    return instances_train, instances_dev, instances_test


def build_optimizer_w_lr_mul(model_param_optimizer, learning_rate, weight_decay):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # Prepare optimizer
    param_optimizer = model_param_optimizer

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay,
         'lr': learning_rate},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': learning_rate}]
    return optimizer_grouped_parameters


def setup_optimizer(model, opts):
    """model_type: str, one of [transformer, cnn]"""

    bert_param_optimizer = [
        (n, p) for n, p in list(model.named_parameters())
        if "bert" in n and p.requires_grad]
    pgp_param_optimizer = [
        (n, p) for n, p in list(model.named_parameters())
        if (opts.model_prefix in n or 'lstm' in n) and p.requires_grad]

    # print(opts.model_prefix)
    # print("bert_param_optimizer", bert_param_optimizer)
    # print("pgp_param_optimizer", pgp_param_optimizer)

    bert_grouped_parameters = build_optimizer_w_lr_mul(
        bert_param_optimizer, opts.bert_learning_rate, opts.bert_weight_decay)
    pgp_grouped_parameters = build_optimizer_w_lr_mul(
        pgp_param_optimizer, opts.learning_rate, opts.weight_decay)

    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.extend(bert_grouped_parameters)
    optimizer_grouped_parameters.extend(pgp_grouped_parameters)
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas)
    return optimizer


dev_best_f1 = 0
best_f1 = 0
flag = 0


@torch.no_grad()
def validate(model, val_loader, cfg, train_global_step, mode='dev', model_saver=None):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    LOGGER.info('*' * 20 + f"The performance on {mode} set" + '*' * 20)

    model.eval()
    st = time.time()
    debug_step = 5
    global dev_best_f1, best_f1, flag

    all_true_rev_args_list = []
    all_pred_rev_args_list = []
    all_true_rep_args_list = []
    all_pred_rep_args_list = []
    all_true_arg_pairs_list = []
    all_pred_arg_pairs_list = []
    threshold_pred_arg_pairs_list = []
    for val_step, batch_i in enumerate(range(val_loader.batch_count)):
        review_bert_tokens, reply_bert_tokens, \
        review_num_tokens, reply_num_tokens, \
        review_lengths, reply_lengths, \
        review_masks, reply_masks, \
        rev2rev_weights, rev2rep_weights, \
        rep2rev_weights, rep2rep_weights, \
        review_adjs, reply_adjs, \
        pair_matrix, \
        mask_sent_list,\
        review_ibose_list, reply_ibose_list, \
        rev_arg_2_rep_arg_tags_list, rep_arg_2_rev_arg_tags_list, \
        rev_arg_2_rep_arg_list, rep_arg_2_rev_arg_list = val_loader.get_batch(batch_i)

        pred_rev_args_list, pred_rep_args_list, \
        pred_args_pair_dict_list, pred_args_pair_2_dict_list = model(review_bert_tokens, reply_bert_tokens,
                                                                     review_num_tokens, reply_num_tokens,
                                                                     review_lengths, reply_lengths,
                                                                     review_masks, reply_masks,
                                                                     rev2rev_weights, rev2rep_weights,
                                                                     rep2rev_weights, rep2rep_weights,
                                                                     review_adjs, reply_adjs,
                                                                     pair_matrix,
                                                                     mask_sent_list,
                                                                     review_ibose_list, reply_ibose_list,
                                                                     rev_arg_2_rep_arg_tags_list,
                                                                     rep_arg_2_rev_arg_tags_list,
                                                                     mode="val")

        true_rev_args_list = extract_arguments(review_ibose_list)
        all_true_rev_args_list.extend(true_rev_args_list)
        all_pred_rev_args_list.extend(pred_rev_args_list)

        true_rep_args_list = extract_arguments(reply_ibose_list)
        all_true_rep_args_list.extend(true_rep_args_list)
        all_pred_rep_args_list.extend(pred_rep_args_list)

        true_arg_pairs_list = []
        for rev_arg_2_rep_arg in rev_arg_2_rep_arg_list:
            arg_pairs = []
            for rev_arg, rep_args in rev_arg_2_rep_arg.items():
                for rep_arg in rep_args:
                    arg_pairs.append((rev_arg, rep_arg))
            true_arg_pairs_list.append(arg_pairs)
        all_true_arg_pairs_list.extend(true_arg_pairs_list)

        pred_arg_pairs_list = []
        threshold_arg_pairs_list = []
        for pred_rep_args in pred_args_pair_dict_list:
            pred_arg_pairs = []
            threshold_arg_pairs = {}
            for rev_arg, rep_args in pred_rep_args.items():
                for rep_arg, rep_arg_prob in zip(rep_args[0], rep_args[1]):
                    pred_arg_pairs.append((rev_arg, rep_arg))
                    threshold_arg_pairs[(rev_arg, rep_arg)] = rep_arg_prob
            pred_arg_pairs_list.append(pred_arg_pairs)
            threshold_arg_pairs_list.append(threshold_arg_pairs)

        pred_arg_pairs_2_list = []
        threshold_arg_pairs_2_list = []
        for pred_rep_args_2 in pred_args_pair_2_dict_list:
            pred_arg_pairs = []
            threshold_arg_pairs = {}
            for rep_arg, rev_args in pred_rep_args_2.items():
                for rev_arg, rev_arg_prob in zip(rev_args[0], rev_args[1]):
                    pred_arg_pairs.append((rev_arg, rep_arg))
                    threshold_arg_pairs[(rev_arg, rep_arg)] = rev_arg_prob
            pred_arg_pairs_2_list.append(pred_arg_pairs)
            threshold_arg_pairs_2_list.append(threshold_arg_pairs)

        for r_1_args, r_2_args in zip(threshold_arg_pairs_list, threshold_arg_pairs_2_list):
            pair_set = set(r_1_args.keys()) & set(r_2_args.keys())
            for pair, p in r_1_args.items():
                if p > cfg.pair_threshold:
                    pair_set.add(pair)
            for pair, p in r_2_args.items():
                if p > cfg.pair_threshold:
                    pair_set.add(pair)
            threshold_pred_arg_pairs_list.append(list(pair_set))

        all_pred_arg_pairs_list.extend([list(set(a + b)) for a, b in zip(pred_arg_pairs_list, pred_arg_pairs_2_list)])

    rev_dict = args_metric(all_true_rev_args_list, all_pred_rev_args_list)
    args_pair_dict = args_metric(all_true_arg_pairs_list, all_pred_arg_pairs_list)
    threshold_args_pair_dict = args_metric(all_true_arg_pairs_list, threshold_pred_arg_pairs_list)

    rep_dict = args_metric(all_true_rep_args_list, all_pred_rep_args_list)

    am_dict = args_metric(all_true_rev_args_list + all_true_rep_args_list,
                          all_pred_rev_args_list + all_pred_rep_args_list)

    enhanced_all_pred_rep_args_list = []
    for idx, arg_pairs in enumerate(all_pred_arg_pairs_list):
        pred_rep_args = list(set([arg_pair[1] for arg_pair in arg_pairs]))
        enhanced_all_pred_rep_args_list.append(list(set(all_pred_rep_args_list[idx] + pred_rep_args)))

    enhanced_rep_dict = args_metric(all_true_rep_args_list, enhanced_all_pred_rep_args_list)
    enhanced_am_dict = args_metric(all_true_rev_args_list + all_true_rep_args_list,
                                   all_pred_rev_args_list + enhanced_all_pred_rep_args_list)

    LOGGER.info('rev ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
        rev_dict['f1'], rev_dict['pre'], rev_dict['rec']))
    LOGGER.info('rep ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
        rep_dict['f1'], rep_dict['pre'], rep_dict['rec']))
    LOGGER.info('am ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
        am_dict['f1'], am_dict['pre'], am_dict['rec']))
    LOGGER.info('pair f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
        args_pair_dict['f1'], args_pair_dict['pre'], args_pair_dict['rec']))

    total_f1 = (am_dict['f1'] + args_pair_dict['f1']) / 2

    if mode == "dev" and total_f1 > dev_best_f1:
        dev_best_f1 = total_f1
        flag = 1
    if mode == "test" and falg == 1:
        best_f1 = total_f1
        # if model_saver != None:
        #     model_saver.save(step=0, model=model)
            # model_saver.save(step=train_global_step, model=model)

    LOGGER.info('BEST f1: {:.4f}'.format(best_f1))

    LOGGER.info(f"{mode} finished in {int(time.time() - st)} seconds.")
    model.train()


def start_training(cfg):
    set_random_seed(cfg.seed)

    special_tokens = ['[ENDL]', '[TAB]', '[LINE]', '[EQU]', '[URL]', '[NUM]', '[SPE]']
    tokenizer = context_models[cfg.model_class]['tokenizer'].from_pretrained(cfg.bert_weights_path,
                                                                         additional_special_tokens=special_tokens)

    # prepare data
    instances_train, instances_dev, instances_test = setup_dataloaders(cfg, tokenizer)

    trainset = DataIterator(instances_train, cfg, cfg.train_batch_size)
    devset = DataIterator(instances_dev, cfg, cfg.val_batch_size)
    testset = DataIterator(instances_test, cfg, cfg.val_batch_size)

    # compute the number of steps and update cfg
    total_n_examples = len(trainset)
    total_train_batch_size = int(cfg.train_batch_size * cfg.gradient_accumulation_steps)
    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))
    cfg.num_warmup_steps = int(cfg.num_train_steps * cfg.warmup_ratio)

    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    # setup model and optimizer

    # cfg.ntm_model = cfg.ntm_model + f"ntm_model_{cfg.topic_num}.pt"
    cfg.ntm_model = cfg.ntm_model + f"ntm_model_intra_CLS_inter_ALL_sent_top_3_intra_neighbour_1_" \
                                    f"iw_weight_norm_{cfg.iw_weight_norm}_word_top_{cfg.word_top}.pt"
    model = setup_model(cfg)
    model.train()

    optimizer = setup_optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.num_warmup_steps,
                                                num_training_steps=cfg.num_train_steps)

    # restore
    now_time = datetime.datetime.now()
    now_time = now_time.strftime("%Y-%m-%d-%X-%a")
    savePath = join(cfg.output_dir, now_time)
    cfg.output_dir = savePath  # if you need restore your checkpoint, please annotate here and use the correct outdir in order line

    # restore
    restorer = E2E_TrainingRestorer(cfg, model=model, optimizer=optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step

    LOGGER.info("Saving training meta...")
    save_training_meta(cfg)
    LOGGER.info("Saving training done...")

    TB_LOGGER.create(join(cfg.output_dir, 'log'))
    pbar = tqdm(total=cfg.num_train_steps)
    model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
    add_log_to_file(join(cfg.output_dir, "log", "log.txt"))

    # torch.save(model.state_dict(), join(cfg.output_dir, "model_init.pt"))

    if global_step > 0:
        pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"Accumulate steps = {total_train_batch_size}")
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate every {cfg.valid_steps} steps, in total {actual_num_valid} times")

    debug_step = 3
    running_am_loss = RunningMeter('train_am_loss')
    running_ape_loss = RunningMeter('train_ape_loss')
    running_cl_loss = RunningMeter('train_cl_loss')
    step = 0
    for epoch in range(cfg.num_train_epochs):
        LOGGER.info(f'Start training in epoch: {epoch}')
        # for batch in InfiniteIterator(train_loader):

        for batch_i in range(trainset.batch_count):

            n_epoch = int(1. * total_train_batch_size * global_step / total_n_examples)
            # LOGGER.info("Running epoch: {}".format(n_epoch))

            review_bert_tokens, reply_bert_tokens, \
            review_num_tokens, reply_num_tokens, \
            review_lengths, reply_lengths, \
            review_masks, reply_masks, \
            rev2rev_weights, rev2rep_weights,\
            rep2rev_weights, rep2rep_weights,\
            review_adjs, reply_adjs, \
            pair_matrix, \
            mask_sent_list, \
            review_ibose_list, reply_ibose_list, \
            rev_arg_2_rep_arg_tags_list, rep_arg_2_rev_arg_tags_list, \
            rev_arg_2_rep_arg_list, rep_arg_2_rev_arg_list = trainset.get_batch(batch_i)

            # loss_link, \
            crf_loss, pair_loss, cl_loss = model(review_bert_tokens, reply_bert_tokens,
                                                 review_num_tokens, reply_num_tokens,
                                                 review_lengths, reply_lengths,
                                                 review_masks, reply_masks,
                                                 rev2rev_weights, rev2rep_weights,
                                                 rep2rev_weights, rep2rep_weights,
                                                 review_adjs, reply_adjs,
                                                 pair_matrix,
                                                 mask_sent_list,
                                                 review_ibose_list, reply_ibose_list,
                                                 rev_arg_2_rep_arg_tags_list, rep_arg_2_rev_arg_tags_list
                                                 )

            if cfg.use_ntm and epoch < cfg.warm_up_ntm:
                ntm_loss = torch.tensor(0).to(cfg.device)

            loss = cfg.pair_weight * pair_loss + crf_loss + cfg.cl_weight * cl_loss
            # print("loss", loss)

            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

            loss.backward()

            running_am_loss(crf_loss.item())
            running_ape_loss(pair_loss.item())
            running_cl_loss(cl_loss.item())

            # backward pass
            # optimizer
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                TB_LOGGER.add_scalar('train/am_loss', running_am_loss.val, global_step)
                TB_LOGGER.add_scalar('train/ape_loss', running_ape_loss.val, global_step)
                TB_LOGGER.add_scalar('train/cl_loss', running_cl_loss.val, global_step)

                # update model params
                if cfg.grad_norm != -1:
                    grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_norm)
                    TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)

                # Check if there is None grad
                # none_grads = [
                #     p[0] for p in model.named_parameters()
                #     if p[1].requires_grad and p[1].grad is None]
                # print(len(none_grads), none_grads)
                # assert len(none_grads) == 2, f"{none_grads}"

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                restorer.step()
                pbar.update(1)

                # print(len(optimizer.param_groups))
                assert len(optimizer.param_groups) == 4
                for pg_n, param_group in enumerate(optimizer.param_groups):
                    if pg_n == 0:
                        lr_this_step_transformer = param_group['lr']
                    elif pg_n == 2:
                        lr_this_step_ptg = param_group['lr']

                TB_LOGGER.add_scalar(
                    "train/lr_bert", lr_this_step_transformer, global_step)
                TB_LOGGER.add_scalar(
                    "train/lr_ptg", lr_this_step_ptg, global_step)

                # print("train/lr_bert", lr_this_step_transformer)
                # print("train/lr_pgp", lr_this_step_pgp)

                TB_LOGGER.step()

                # checkpoint
                if global_step % cfg.valid_steps == 0:
                    LOGGER.info(f'Step {global_step}: start validation in epoch: {n_epoch}')
                    validate(model, devset, cfg, global_step, 'dev')
                    validate(model, testset, cfg, global_step, 'test', model_saver)
            # model_saver.save(step=global_step, model=model)
            if global_step >= cfg.num_train_steps:
                break

            if cfg.debug and global_step >= debug_step:
                break

            step += 1

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, devset, cfg, global_step, 'dev')
        validate(model, testset, cfg, global_step, 'test')
    # model_saver.save(step=global_step, model=model)


if __name__ == '__main__':
    cfg = shared_configs.get_args()
    start_training(cfg)
