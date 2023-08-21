"""
Modified from UNITER code
"""
import os
import sys
import json
import argparse

from easydict import EasyDict as edict


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
    (1) it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    (2) it also sets configs in the config file but declared in the parser
    """
    # convert to EasyDict object, enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    # del args.config
    return args


def str2bool(v):
    if isinstance(v, bool) and v == True:
        return True
    if isinstance(v, bool) and v == False:
        return False
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class SharedConfigs(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self, desc="shared config for pretraining and finetuning"):
        parser = argparse.ArgumentParser(description=desc)
        # debug parameters
        parser.add_argument(
            "--debug", type=int, choices=[0, 1], default=0,
            help="debug mode, output extra info & break all loops."
                 "0: disable, 1 enable")
        parser.add_argument(
            "--data_ratio", type=float, default=1.0,
            help="portion of train/val exampels to use,"
                 "e.g., overfit a small set of data")
        parser.add_argument("--gpu", type=int)

        # Required parameters
        parser.add_argument(
            "--model_config", type=str,
            help="path to model structure config json")
        parser.add_argument(
            "--tokenizer_dir", type=str, help="path to tokenizer dir")
        parser.add_argument(
            "--output_dir", type=str,
            help="dir to store model checkpoints & training meta.")

        # data preprocessing parameters
        parser.add_argument(
            "--max_bert_token", type=int, default=200, help="max text #tokens ")
        parser.add_argument(
            "--encoding_scheme", type=str, default="IOBES", help="BIO, IOBES")
        parser.add_argument(
            "--num_instances", type=int, default=10, help="-1: all instances")
        
        
        # training parameters
        parser.add_argument(
            "--train_batch_size", default=32, type=int,
            help="Single-GPU batch size for training for Horovod.")
        parser.add_argument(
            "--val_batch_size", default=128, type=int,
            help="Single-GPU batch size for validation for Horovod.")
        parser.add_argument(
            "--gradient_accumulation_steps", type=int, default=1,
            help="#updates steps to accumulate before performing a backward/update pass."
                 "Used to simulate larger batch size training. The simulated batch size "
                 "is train_batch_size * gradient_accumulation_steps for a single GPU.")
        parser.add_argument("--learning_rate", default=1e-3, type=float,
                            help="initial learning rate.")
        parser.add_argument(
            "--num_valid", default=20, type=int,
            help="Run validation X times during training and checkpoint.")
        parser.add_argument(
            "--min_valid_steps", default=100, type=int,
            help="minimum #steps between two validation runs")
        parser.add_argument(
            "--save_steps_ratio", default=0.01, type=float,
            help="save every 0.01*global steps to resume after preemption,"
                 "not used for checkpointing.")
        parser.add_argument("--num_train_epochs", default=50, type=int,
                            help="Total #training epochs.")
        parser.add_argument("--optim", default="adamw",
                            choices=["adam", "adamax", "adamw"],
                            help="optimizer")
        parser.add_argument("--betas", default=[0.9, 0.98],
                            nargs=2, help="beta for adam optimizer")
        parser.add_argument("--decay", default="linear",
                            choices=["linear", "invsqrt"],
                            help="learning rate decay method")
        parser.add_argument("--dropout", default=0.1, type=float,
                            help="tune dropout regularization")
        parser.add_argument("--weight_decay", default=1e-3, type=float,
                            help="weight decay (L2) regularization")
        parser.add_argument("--grad_norm", default=1.0, type=float,
                            help="gradient clipping (-1 for no clipping)")
        parser.add_argument(
            "--warmup_ratio", default=0.1, type=float,
            help="to perform linear learning rate warmup for. (invsqrt decay)")
        parser.add_argument("--lr_mul", default=1.0, type=float,
                            help="lr_mul for model")
        parser.add_argument(
            "--lr_mul_prefix", default="", type=str, help="lr_mul param prefix for model")
        parser.add_argument("--lr_decay", default="linear", choices=["linear", "invsqrt", "multi_step", "constant"],
                            help="learning rate decay method")
        parser.add_argument("--step_decay_epochs", type=int,
                            nargs="+", help="multi_step decay epochs")
        # bert parameters
        parser.add_argument("--bert_optim", default="adamw", type=str,
                            choices=["adam", "adamax", "adamw", "sgd"],
                            help="optimizer for Bert")
        parser.add_argument("--bert_learning_rate", default=1e-5, type=float,
                            help="learning rate for Bert")
        parser.add_argument("--bert_weight_decay", default=1e-3, type=float,
                            help="weight decay for Bert")
        parser.add_argument("--sgd_momentum", default=0.9, type=float,
                            help="momentum for Bert")
        parser.add_argument("--bert_lr_mul", default=1.0, type=float,
                            help="lr_mul for Bert")
        parser.add_argument(
            "--bert_lr_mul_prefix", default="grid_encoder", type=str,
            help="lr_mul param prefix for Bert")
        parser.add_argument("--bert_lr_decay", default="linear",
                            choices=["linear", "invsqrt", "multi_step",
                                     "constant"],
                            help="learning rate decay method")
        parser.add_argument("--bert_step_decay_epochs", type=int,
                            nargs="+", help="Bert multi_step decay epochs")
        parser.add_argument(
            "--freeze_bert", default=0, choices=[0, 1], type=int,
            help="freeze Bert by setting the requires_grad=False for Bert parameters.")

        parser.add_argument(
            "--model_prefix", default="ptg", type=str,
            help="model_prefix for attmodel or pgpmodel")
        parser.add_argument("--num_layers", default=1, type=int, help="the number of layers in lstm")
        parser.add_argument("--bidirect", default=1, type=int, choices=[0, 1], help="whether bidirect for lstm or not")
        
        # model arch # checkpoint
        parser.add_argument("--e2e_weights_path", type=str,
                            help="path to e2e model weights")
        parser.add_argument("--bert_weights_path", type=str, default="/home/data/bert-base-uncased",
                            help="path to BERT weights, only use for finetuning")
        
        # model parameters
        
        parser.add_argument("--hidden_size", default=128, type=int,
                            help="hidden size for model.")
        parser.add_argument("--bert_output_size", default=768, type=int,
                            help="bert_output_size for model.")
        parser.add_argument("--bi_num_attention_heads", default=4, type=int,
                            help="bi_num_attention_heads for dual cross attention in model.")
        parser.add_argument("--num_attention_heads", default=4, type=int,
                            help="num_attention_heads for focus-attention, cross-attention and self-attention in model.")
        parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                            help="attention_probs_dropout_prob for dropout attention probs in model.")

        parser.add_argument("--gcn_layer", type=int, help="")
        parser.add_argument("--gat_layer", type=int, help="")
        
        parser.add_argument("--use_pos_emb", type=str2bool, default='1')
        parser.add_argument("--graph_type", type=str, default="single", help="single or double")
        parser.add_argument("--aggregation_type", default="one_stage", type=str, help="one_stage or two_stage")
        parser.add_argument("--tg_layer_num", type=int, default=1, help="")
        parser.add_argument("--cross_type", default="concat", type=str, help="concat or dot")
        parser.add_argument("--graph_residual", type=str2bool, default='0', help="")
        
        parser.add_argument('--attention_loss', type=bool, default=False)
        parser.add_argument("--att_loss_type", default="", type=str, help="cross or self or gate or all, '' means all")
        parser.add_argument("--gate_loss_type", default="", type=str, help="2, 3")
        parser.add_argument("--self_loss_type", default="", type=str, help="2_1, 2_2, 3")

        parser.add_argument("--ape_in_type", default="", type=str, help="1, 2, 3")
        
        parser.add_argument('--attn_weight', type=float, default=1)

        parser.add_argument('--pair_weight', type=float, default=0.6,
                            help='pair loss weight coefficient for loss computation')
        parser.add_argument('--nl_weight', type=float, default=0.01,
                            help='ntm loss weight coefficient for loss computation')

        parser.add_argument("--token_embedding", type=bool, default=True)
        parser.add_argument("--num_embedding_layer", type=int, default=1)

        parser.add_argument("--use_sent_rep", type=str2bool, default='1')
        parser.add_argument("--use_probing_topic", type=str2bool, default='0')
        parser.add_argument("--use_ntm", type=str2bool, default='0')
        parser.add_argument("--use_topic_project", type=str2bool, default='0')
        parser.add_argument("--warm_up_ntm", type=int, default=-1,
                            help='if -1, donot load ntm and training ntm without ntm_loss in the top {warm_up_ntm} epoch ')
        parser.add_argument("--ntm_model", type=str, default='', help='the ntm model path')
        
        parser.add_argument('--share_crf_param', type=bool, default=True,
                            help='whether to share same CRF layer for review&reply decoding')
        parser.add_argument('--negative_sample', type=int, default=1000,
                            help='number of negative samples, 1000 means all')
        parser.add_argument('--pair_threshold', type=float, default=0.5,
                            help='pairing threshold during evaluation')
        parser.add_argument('--ema', type=float, default=1.0,
                            help='EMA coefficient alpha')
        parser.add_argument('--gamma', type=float, default=1,
                            help='gamma in kernel function for calculating relevant position')

        

        # probing topic parameters
        parser.add_argument('--metric', default='dist',
                            help='metrics for impact calculation, support [dist, cos] so far')
        parser.add_argument(
            "--include_iw_num0", type=str2bool, default='0', help="whether to save the sentence with iw_num=0")
        
        parser.add_argument('--sent_top', default=3, type=int, help="tok k relevant sentences")
        parser.add_argument('--intra_neighbour', default=3, type=int, help="tok k relevant sentences")
        
        parser.add_argument('--intra_type', default='CLS', type=str, help="CLS, IW, OW, OW2, OW3, OW4")
        parser.add_argument('--inter_type', default='ALL', type=str, help="CLS, ALL")
        parser.add_argument('--iw_weight_norm', type=str2bool, default='1',
                            help='whether to normalize the weight from intra- and inter-sentences perspective')
        
        parser.add_argument('--word_top', default=10, type=int, help="tok k import word")
        parser.add_argument('--probing_layers', default=1, type=int, help="tok layers to be probed")
        parser.add_argument('--probing_dataset', default='train', type=str, help="train, dev, test")
        
        # NTM
        parser.add_argument('--topic_num', type=int, default=50)
        parser.add_argument('--ntm_warm_up_epochs', type=int, default=200)
        parser.add_argument('--target_sparsity', type=float, default=0.85)
        
        
        # probing graph weight
        parser.add_argument('--probing_mask', default='twomask', type=str, help="onemask, twomask")
        parser.add_argument('--norm_score', default='sample', type=str, help="sent, sample")
        parser.add_argument('--start_sample', default=-1, type=int, help="")
        parser.add_argument('--max_cor_num_type', default='sample', type=str, help="sent, sample, all(all smaple)")
        parser.add_argument('--poincare_threshold', default=0.7, type=float, help="")
        parser.add_argument('--dist_threshold', default=0.5, type=float, help="edui")
        parser.add_argument('--intra_cor_threshold', default=1, type=int, help="")
        parser.add_argument('--inter_cor_threshold', default=2, type=int, help="")
        parser.add_argument('--adj_norm', default='non_symmetric_normalize_adj', type=str,
                            help="non_symmetric_normalize_adj2, non_symmetric_normalize_adj")
        parser.add_argument('--noone', default='no1', type=str, help=" '' or no1 ")


        #span2span
        parser.add_argument('--span_extractor_type', default='endpoint', type=str, help="endpoint, mean, attn")
        parser.add_argument('--use_span_width_embed', type=str2bool, default='1')
        parser.add_argument('--max_span_width', default=10, type=int, help="")
        parser.add_argument('--use_biaffine_classifier', type=str2bool, default='0')
        parser.add_argument('--pair_rep_type', default='cat_dot', type=str, help="endpoint, mean, attn")
        parser.add_argument('--use_rel_score', type=str2bool, default='0')
        parser.add_argument('--use_prune_sent', type=str2bool, default='1')
        parser.add_argument('--spans_per_sample', default=0.5, type=float, help="")
        parser.add_argument('--use_gold_for_train_prune_scores', type=str2bool, default='0')
        parser.add_argument('--use_prune_span', type=str2bool, default='0')


        # cl
        parser.add_argument('--use_cl_loss', default='1', type=str2bool, help="")
        parser.add_argument('--cl_type', default='', type=str, help="sup_cl, bpr_all, bpr_cross, sub_cl")
        parser.add_argument('--sim_type', default='', type=str, help="cos, dot")
        parser.add_argument('--use_only_sents_in_ac', default='1', type=str2bool, help="")
        parser.add_argument('--sub_type', default='', type=str, help="pre, post")
        parser.add_argument('--temperature', default=0.07, type=float, help="")
        parser.add_argument('--cl_weight', default=1, type=float, help="")
        parser.add_argument('--use_cl_proj', default='0', type=str2bool, help="")
        parser.add_argument('--cl_layer', default='', type=str, help="bert, graph")
        parser.add_argument('--rep_type', default='', type=str, help="mean, cls")
        parser.add_argument('--cl_proj_dim', default=0, type=int, help="")
        parser.add_argument('--drop_edge_type', default='', type=str, help="uniform, bigger, smaller")
        parser.add_argument('--drop_edge_rate', default=1, type=float, help="")
        parser.add_argument('--drop_type', default='', type=str, help="")


        # compare wordmet amd word2vec
        parser.add_argument('--graph_postfix', default='', type=str, help="graph_postfix")


        # inference only, please include substring `inference'
        # in the option to avoid been overwrite by loaded options,
        # see start_inference() in run_vqa_w_hvd.py
        parser.add_argument("--inference_model_step", default=-1, type=int,
                            help="pretrained model checkpoint step")
        parser.add_argument(
            "--do_inference", default=0, type=int, choices=[0, 1],
            help="perform inference run. 0: disable, 1 enable")
        parser.add_argument(
            "--inference_split", default="val",
            help="For val, the data should have ground-truth associated it."
                 "For test*, the data comes with no ground-truth.")
        parser.add_argument("--data_path", type=str, default="/home/PTG4AM/data/RR-submission-v2",
                            help="path to data file for train")
        parser.add_argument("--split_test_file_path", type=str,
                            help="path to data file for test")
        parser.add_argument("--vocab_path", default="/home/PTG4AM/data/bow_vocab.txt", type=str)
        
        # device parameters
        parser.add_argument('--device', type=str, default="cuda",
                            help='cuda or cpu')
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument(
            "--fp16", type=int, choices=[0, 1], default=0,
            help="Use 16-bit float precision instead of 32-bit."
                 "0: disable, 1 enable")
        parser.add_argument("--n_workers", type=int, default=4,
                            help="#workers for data loading")
        parser.add_argument("--pin_mem", type=int, choices=[0, 1], default=1,
                            help="pin memory. 0: disable, 1 enable")

        parser.add_argument("--model_version", type=str, default='1')

        # can use config files, will only overwrite unset parameters
        parser.add_argument("--config", help="JSON config files")
        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        args = parse_with_config(parsed_args)

        # convert to all [0, 1] options to bool, including these task specific ones
        zero_one_options = [
            "fp16", "pin_mem", "debug", "do_inference", "bidirect",
        ]
        for option in zero_one_options:
            if hasattr(args, option):
                setattr(args, option, bool(getattr(args, option)))

        # basic checks
        # This is handled at TrainingRestorer
        # if exists(args.output_dir) and os.listdir(args.output_dir):
        #     raise ValueError(f"Output directory ({args.output_dir}) "
        #                      f"already exists and is not empty.")
        if args.bert_step_decay_epochs and args.bert_lr_decay != "multi_step":
            Warning(
                f"--bert_step_decay_epochs set to {args.bert_step_decay_epochs}"
                f"but will not be effective, as --bert_lr_decay set to be {args.bert_lr_decay}")
        if args.step_decay_epochs and args.decay != "multi_step":
            Warning(
                f"--step_decay_epochs epochs set to {args.step_decay_epochs}"
                f"but will not be effective, as --decay set to be {args.decay}")

        assert args.gradient_accumulation_steps >= 1, \
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps} "

        assert 1 >= args.data_ratio > 0, \
            f"--data_ratio should be [1.0, 0), but get {args.data_ratio}"

        return args
    
    def get_args(self):
        args = self.parse_args()
    
        return args
    
shared_configs = SharedConfigs()
