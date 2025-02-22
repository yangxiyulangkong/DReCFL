# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from fl_library.algorithms.servers.serveravg import FedAvg
from fl_library.algorithms.servers.server_DReCFL import DReCFL
from fl_library.trainmodel.models import *

from fl_library.utils.result_utils import average_data
from fl_library.utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr": # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn": # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "dnn": # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        # elif model_str == "resnet":
        #     args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        #
        #     # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
        #     # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        #     # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        #
        #     # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        #
        # elif model_str == "resnet10":
        #     args.model = resnet10(num_classes=args.num_classes).to(args.device)
        #
        # elif model_str == "resnet34":
        #     args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)
        #
        # elif model_str == "alexnet":
        #     args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
        #
        #     # args.model = alexnet(pretrained=True).to(args.device)
        #     # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        #     # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        #
        # elif model_str == "googlenet":
        #     args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
        #                                               num_classes=args.num_classes).to(args.device)
        #
        #     # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
        #     # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        #     # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        #
        # elif model_str == "mobilenet_v2":
        #     args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
        #
        #     # args.model = mobilenet_v2(pretrained=True).to(args.device)
        #     # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        #     # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        #
        # elif model_str == "lstm":
        #     args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)
        #
        # elif model_str == "bilstm":
        #     args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
        #                                            output_size=args.num_classes, num_layers=1,
        #                                            embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
        #                                            embedding_length=emb_dim).to(args.device)
        #
        # elif model_str == "fastText":
        #     args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)
        #
        # elif model_str == "TextCNN":
        #     args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
        #                          num_classes=args.num_classes).to(args.device)
        #
        # elif model_str == "Transformer":
        #     args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, nlayers=2,
        #                                   num_classes=args.num_classes, max_len=max_len).to(args.device)
        #
        # elif model_str == "AmazonMLP":
        #     args.model = AmazonMLP().to(args.device)
        #
        # elif model_str == "harcnn":
        #     if args.dataset == 'HAR':
        #         args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
        #                             pool_kernel_size=(1, 2)).to(args.device)
        #     elif args.dataset == 'PAMAP2':
        #         args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
        #                             pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        elif args.algorithm == "DReCFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = DReCFL(args, i)
 
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":

    total_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=50)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=300)
    parser.add_argument('-ls', "--local_epochs", type=int, default=10,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="DReCFL")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.2,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
