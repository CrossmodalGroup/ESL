from logging import fatal
import os
from pickle import TRUE
from lib import evaluation

import torch
torch.set_num_threads(4)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


## for heuristic_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 44 of ves.py as True
# RUN_PATH = "../checkpoint_heuristic_mscoco_bert.tar"


## for adaptive_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 44 of ves.py as False
RUN_PATH = "../checkpoint_adaptive_mscoco_bert.tar"

DATA_PATH = "../MS-COCO/"

### set fold5 as Flase for 5K TEST, otherwise for AVERAGE 1K TEST
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="testall", fold5=False)
