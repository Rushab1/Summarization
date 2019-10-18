#cd ../preprocess/
#python gigaword.py -max_files 5000
#cd ../Code/

echo "\n\n\n===================================="
echo "STARTING compute_sent_similarity"
echo "====================================\n\n"
#python compute_sent_similarity.py --dataset gigaword --test_split 0.8 --val_split 0.1

echo "\n\n\n===================================="
echo "STARTING val_test_predictions"
echo "====================================\n\n"
#python ./val_test_predictions.py -dataset gigaword --force_create_embeddings --force_new_predictions -parallelism 100
python ./val_test_predictions.py -dataset gigaword -parallelism 100

