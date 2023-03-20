import argparse
import os
import numpy as np

from src.handlers.evaluater import Evaluater


if __name__ == '__main__':
    ### Decoding arguments
    eval_parser = argparse.ArgumentParser(description='Arguments for training the system')
    eval_parser.add_argument('--path', type=str, help='path to experiment')
    eval_parser.add_argument('--dataset', type=str, help='dataset to train the system on')
    eval_parser.add_argument('--mode', default='test', type=str, help='which data split to evaluate on')
    eval_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')
    eval_parser.add_argument('--lim', type=int, default=None, help='whether subset of data to be used') 
    args = eval_parser.parse_args()

    evaluater = Evaluater(args.path, args.device)
    pred_texts = evaluater.load_pred_texts(args.dataset, args.mode)
    label_texts = evaluater.load_label_texts(args.dataset, args.mode)
    evaluator.calculate_rouge(pred_texts, label_texts, display=True)
    evaluator.calculate_bleu(pred_texts, label_texts, display=True)
