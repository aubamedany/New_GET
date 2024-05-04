import sys
sys.path.insert(0, '../../GET')
sys.path.insert(0, '../GET')
import datetime
# import tensorflow
from Models.FCWithEvidences import graph_based_semantic_structure
from Fitting.FittingFC import char_man_fitter_query_repr1
import time
import json
from interactions import ClassificationInteractions
import matchzoo as mz
from handlers import cls_load_data
import argparse
import random
import numpy as np
import torch
import os
import datetime
from handlers.output_handler_FC import FileHandlerFC
from Evaluation import mzEvaluator as evaluator
from setting_keywords import KeyWordSettings
from matchzoo.embedding import entity_embedding

def fit_models(args):
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    curr_date = datetime.datetime.now().timestamp()  # seconds
    # folder to store all outputed files of a run
    secondary_log_folder=""
    if(args.use_oc):
        secondary_log_folder = os.path.join(args.log, f"log_results_{args.dataset}_{args.lamda1}_{args.batch_size}_OC_LOSS")
    else:
        secondary_log_folder = os.path.join(args.log, f"log_results_{args.dataset}_{args.lamda1}")
    if not os.path.exists(secondary_log_folder):
        os.mkdir(secondary_log_folder)
    args.secondary_log_folder = secondary_log_folder
    # args.seed = random.randint(1, 150000)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    root = os.path.join(os.path.join(args.path, args.dataset), "mapped_data")
    tx = time.time()
    kfold_dev_results, kfold_test_results = [], []
    list_metrics = KeyWordSettings.CLS_METRICS                  # evaluation metrics
    for i in range(args.num_folds):
        outfolder_per_fold = os.path.join(secondary_log_folder, "Fold_%s" % i)
        if not os.path.exists(outfolder_per_fold):
            os.mkdir(outfolder_per_fold)

        logfolder_result_per_fold = os.path.join(outfolder_per_fold, "result_%s.txt" % int(seed))
        file_handler = FileHandlerFC()
        file_handler.init_log_files(logfolder_result_per_fold)
        settings = json.dumps(vars(args), sort_keys=True, indent=2)
        file_handler.myprint("============= FOLD %s ========================" % i)
        file_handler.myprint(settings)
        file_handler.myprint("Setting seed to " + str(args.seed))
        predict_pack = cls_load_data.load_data(root + "/%sfold" % args.num_folds, 'test_%s' % i, kfolds = args.num_folds)
        train_pack = cls_load_data.load_data(root + "/%sfold" % args.num_folds, 'train_%sres' % i, kfolds = args.num_folds)
        valid_pack = cls_load_data.load_data(root, 'dev', kfolds = args.num_folds)
        # print(train_pack.left.head())

        a = train_pack.left["text_left"].str.lower().str.split().apply(len).max()
        b = valid_pack.left["text_left"].str.lower().str.split().apply(len).max()
        c = predict_pack.left["text_left"].str.lower().str.split().apply(len).max()
        max_query_length = max([a, b, c])
        min_query_length = min([a, b, c])

        a = train_pack.right["text_right"].str.lower().str.split().apply(len).max()
        b = valid_pack.right["text_right"].str.lower().str.split().apply(len).max()
        c = predict_pack.right["text_right"].str.lower().str.split().apply(len).max()
        max_doc_length = max([a, b, c])
        min_doc_length = min([a, b, c])

        file_handler.myprint("Min query length, " + str(min_query_length) + " Min doc length " + str(min_doc_length))
        file_handler.myprint("Max query length, " + str(max_query_length) + " Max doc length " + str(max_doc_length))
        additional_data = {KeyWordSettings.OutputHandlerFactChecking: file_handler,
                           KeyWordSettings.GNN_Window: args.gnn_window_size}
        preprocessor = mz.preprocessors.CharManPreprocessor(fixed_length_left = args.fixed_length_left,
                                                            fixed_length_right = args.fixed_length_right,
                                                            fixed_length_left_src = args.fixed_length_left_src_chars,
                                                            fixed_length_right_src = args.fixed_length_right_src_chars)
        t1 = time.time()
        print('parsing data')
        train_processed = preprocessor.fit_transform(train_pack)  # This is a DataPack
        valid_processed = preprocessor.transform(valid_pack)
        predict_processed = preprocessor.transform(predict_pack)
        # print(train_processed.left.head())

        train_interactions = ClassificationInteractions(train_processed, **additional_data)
        valid_interactions = ClassificationInteractions(valid_processed, **additional_data)
        test_interactions = ClassificationInteractions(predict_processed, **additional_data)

        file_handler.myprint('done extracting')
        t2 = time.time()
        file_handler.myprint('loading data time: %d (seconds)' % (t2 - t1))
        file_handler.myprint("Building model")

        print("Loading word embeddings......")
        t1_emb = time.time()
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        glove_embedding = mz.datasets.embeddings.load_glove_embedding_FC(dimension = args.word_embedding_size,
                                                                         term_index = term_index, **additional_data)

        embedding_matrix = glove_embedding.build_matrix(term_index)
        entity_embs1 = entity_embedding.EntityEmbedding(args.claim_src_emb_size)
        claim_src_embs_matrix = entity_embs1.build_matrix(preprocessor.context['claim_source_unit'].state['term_index'])

        entity_embs2 = entity_embedding.EntityEmbedding(args.article_src_emb_size)
        article_src_embs_matrix = entity_embs2.build_matrix(preprocessor.context['article_source_unit'].state['term_index'])

        t2_emb = time.time()
        print("Time to load word embeddings......", (t2_emb - t1_emb))

        match_params = {}
        match_params['embedding'] = embedding_matrix
        match_params["num_classes"] = args.num_classes
        match_params["fixed_length_right"] = args.fixed_length_right
        match_params["fixed_length_left"] = args.fixed_length_left

        # for claim source
        match_params["use_claim_source"] = args.use_claim_source
        match_params["claim_source_embeddings"] = claim_src_embs_matrix
        # for article source
        match_params["use_article_source"] = args.use_article_source
        match_params["article_source_embeddings"] = article_src_embs_matrix
        # multi-head attention
        match_params["cuda"] = args.cuda
        match_params["num_att_heads_for_words"] = args.num_att_heads_for_words  # first level
        match_params["num_att_heads_for_evds"] = args.num_att_heads_for_evds  # second level

       
        match_params['dropout_gnn'] = args.gnn_dropout
        match_params["dropout_left"] = args.dropout_left
        match_params["dropout_right"] = args.dropout_right
        match_params["hidden_size"] = args.hidden_size

        match_params["gsl_rate"] = args.gsl_rate 

        match_params["embedding_freeze"] = True
        match_params["output_size"] = 2 # if args.dataset == "Snopes" else 3
        match_params["term_index"]= term_index
        match_params["use_transformer"]= args.use_transformer
        match_model = graph_based_semantic_structure.Graph_basedSemantiStructure(match_params)

        file_handler.myprint("Fitting Model")
        fit_model = char_man_fitter_query_repr1.CharManFitterQueryRepr1(net = match_model, loss = args.loss_type, n_iter = args.epochs,
                                                  batch_size = args.batch_size, learning_rate = args.lr,
                                                  early_stopping = args.early_stopping, use_cuda = args.cuda,
                                                  logfolder = outfolder_per_fold, curr_date = curr_date,
                                                  fixed_num_evidences = args.fixed_num_evidences,
                                                  output_handler_fact_checking = file_handler, seed=args.seed,
                                                  output_size=match_params["output_size"],args=args)

        try:
            fit_model.fit(train_interactions, verbose = True,  # for printing out evaluation during training
                          # topN = args.topk,
                          val_interactions=valid_interactions,
                          test_interactions=test_interactions)
            # log_tensor = "logs/tensorboard" +datetime.now().strftime("%Y/%m/%d-%H/%M/%S")
            # tensor_callbacks = tensorflow.keras.callbacks.Tensorboard(log_dir =  log_tensor)
            dev_results, test_results = fit_model.load_best_model(valid_interactions, test_interactions)
            kfold_dev_results.append(dev_results)
            kfold_test_results.append(test_results)
        except KeyboardInterrupt:
            file_handler.myprint('Exiting from training early')
        t10 = time.time()
        file_handler.myprint('Total time for one fold:  %d (seconds)' % (t10 - t1))

    avg_test_results = evaluator.compute_average_classification_results(kfold_test_results, list_metrics, **additional_data)
    file_handler.myprint("Average results from %s folds" % args.num_folds)
    avg_test_results_json = json.dumps(avg_test_results, sort_keys=True, indent=2)
    file_handler.myprint(avg_test_results_json)
    # save results in json file
    result_json_path = os.path.join(secondary_log_folder, "avg_5fold_result_%s.json" % int(seed))
    with open(result_json_path, 'w') as fin:
        fin.write(avg_test_results_json)
    fin.close()
    ty = time.time()
    file_handler.myprint('Total time:  %d (seconds)' % (ty - tx))
    return avg_test_results
