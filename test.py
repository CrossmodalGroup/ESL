import os
from lib import evaluation

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


## for heuristic_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 44 of ves.py as True
# RUN_PATH = "../checkpoint_heuristic_flickr30k_bert.tar"


## for adaptive_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 44 of ves.py as False
RUN_PATH = "../checkpoint_adaptive_flickr30k_bert.tar"


DATA_PATH = "../Flickr30K/"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="test")
