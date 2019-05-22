python translate.py -model ../modelfiles/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt \
    -src ../../train_test_data/untouched_test_data/cleaned_by_sentence_articles.txt \
    -output ../pred/outputs/pred_cleaned_by_sentence_articles_CNN_35.txt \
    -verbose \
    -ignore_when_blocking "." "</t>" "<t>" \
    -min_length 35 \
    -batch_size 20
    #-batch_size 100 \
    #-shard_size 10 \
    #-beam_size 10 \
    #-beta 5 \
    #-stepwise_penalty \
    #-coverage_penalty summary \
    #-length_penalty wu \
    #-alpha 0.9 \
    #-verbose \
    #-replace_unk \
    #-block_ngram_repeat 3 \
