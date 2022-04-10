# CE7455-FinalProject

## How to run

The entry file is main.py. And here are some key arguments:
argument        | remarks
----------------|-----------------------------------------------------------------
subword_vs_list | subword vocabulary size list. "0" means not using subword feature.
char_mode_list  | character embedding list: None or CNN or LSTM. "None" means no character embedding.

In the `main.py` file, it will iterate each subword in `subword_vs_list`, and each char_mode in `char_mode_list`. In other words, it is a for-loop nested in another for-loop. For example, if subword_vs_list is "500 1000", and char_mode_list is "None CNN", then it will train 4 times:
  * subword with vocabulary size 500, and with no character embedding;
  * subword with vocabulary size 500, and with CNN character embedding;
  * subword with vocabulary size 1000, and with no character embedding;
  * subword with vocabulary size 1000, and with CNN character embedding;

Here is sample command to run.
```bash
python -u main.py --data_dir ./data/ --gpu_ids 0 \
	--subword_vs_list 0 200 500 1000 5000 10000 \
	--char_mode_list None CNN \
	--epochs 40 \
	--train_file_list cbt_train.txt \
	--valid_file_list cbt_valid.txt \
	--test_file_list adolescent_valid.txt adult_test.txt \
	--model LSTM \
	--emsize 200 \
	--nhid 200 \
	--nlayers 2 \
	--batch_size 20 \
	--lr 20 \
	--bptt 35
```
