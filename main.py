#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import argparse

import hearst
import utils
import shutil
import joblib
import pandas as pd
from projection.train_Copy import make_embedder, train_model, save_model
from projection.Evaluator_Copy import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd')
    parser.add_argument('--params',help='Parameters for models.')
    parser.add_argument('--dset',help='Corpus path for hypernymy extraction.')
    parser.add_argument('--output',help='Output path for hypernymy pairs.')
    parser.add_argument('--evaluate',help='Evaluation for hypernymy extraction models.')
    args = parser.parse_args()

    if args.cmd == 'hearst':
        hp = hearst.HearstPatterns(extended=args.params)
        if args.dset == None:
            pass
        else:
            data = utils.reader(args.dset)
            pair_list = hp.find_hyponyms(data)
            if args.output is not None:
                utils.writer(args.output,pair_list)
                print('Finished ...')
            else:
                print('Do not specifiy output file, but still show 5 of the results here.')
                for i in range(5):
                    print(pair_list[i])
                print('Finished ...')

    elif args.cmd == 'PPMI':
        # Read prediction from precomputed file
        shutil.copyfile("./patternBased/result_ppmi.txt", args.output)
        with open(args.output) as f:
            print("======== PPMI : score ======== ")
            print("Print out first 5 examples.")
            for i in range(6):
                print(f.readline())
    elif args.cmd == 'PPMISVD':
        path = "result_SvdPpmi_"+str(args.params)+".txt"
        shutil.copyfile("./patternBased/"+path, args.output)
        with open(args.output) as f:
            print("======== PPMI-SVD : score ======== ")
            print("k = ", args.params)
            print("Print out first 5 examples.")
            for i in range(6):
                print(f.readline())

    elif args.cmd == 'dist':
        if args.evaluate is not None:
            res = pd.read_table("./distributional/data/results.txt", sep="\t")
            print("======== Distributional Model : score ======== ")
            print(res.head(6))

    elif args.cmd == 'embed':
        shutil.copyfile("./termEmbed/term_embed_result.txt", args.output)
        if args.evaluate is not None:
            res = pd.read_table("./termEmbed/results.txt", sep="\t")
            print("======== Term Embedding : score ======== ")
            print(res.head(10))

    elif args.cmd == 'Projection':
        if args.output == None:
            pass
        else:
            print("Loading model...\n")
            model = train_model('./projection/dumped.data','./projection/modLog.txt', 9510, use_gpu = False)
            # save_model('./projection/projectMod.pt', model)
            print("Loading test data...\n")
            data=joblib.load('./projection/dumped.data')
            candidates = data["candidates"]
            test_q_cand_ids = data["test_query_cand_ids"]
            test_q_embed = make_embedder(data["test_query_embeds"], grad=False,
                                         cuda=model.use_cuda, sparse=False)
            print("Writing predictions...")
            test_eval = Evaluator(model, test_q_embed, test_q_cand_ids)
            test_eval.write_predictions(args.output, list(candidates))
            print("Done.\n")
        if args.evaluate == 'True':
            Dev_score={}
            Dev_score['DevMAP'] = [];Dev_score['DevAP'] = [];Dev_score['DevMRR'] = []
            for line in open('./projection/modLog.txt',"r"):
                if line.split()[0] == 'Epoch':
                    # header, skip
                    continue
                else:
                    Dev_score['DevMAP'].append(float(line.split()[5]))
                    Dev_score['DevAP'].append(float(line.split()[6]))
                    Dev_score['DevMRR'].append(float(line.split()[7]))
            print("======== Projection Learning : score ======== ")
            print("SemEval2018 Task9.")
            print("DevMAP\tDevAP\tDevMRR")
            print(str(max(Dev_score['DevMAP']))+"\t"+str(max(Dev_score['DevAP']))+"\t"+str(max(Dev_score['DevMRR'])))

    elif args.cmd == 'BiLSTM':
        # use default test data
        if args.output == None:
            pass
        else:
            if args.evaluate is not None:
                print("======== BiLSTM Model : f1 score ======== ")
                with open("./BiLSTM/data/res.score.txt", "r") as score:
                    print(score.read())



if __name__ == '__main__':
    """
        The main function called when main.py is run
        from the command line:

        > python main.py

        See the usage string for more details.

        > python main.py --help
    """
    print('Hypernymy Extraction.')
    print('Please input command <format python main.py [method] [--params] [--dset] [--output] [--evaluate]>.')
    main()
