#!/bin/sh
bash extract_chain_4_10.sh
#matlab -nosplash -nodesktop -r delete_unuseful_col_row_4_10
matlab -nosplash -nodesktop -r delete_unuseful_col_row_8_25
python concatenate_matrix_4_10.py

