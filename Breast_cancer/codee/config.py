import os

ORIG_INPUT_DATASET = "dataset/orig"
BASE_PATH = "dataset/idc"           # new folder which will contain images after computing training & testing split

# training, validation and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.8   # training 80%, testing 20%
VAL_SPLIT = 0.1     # validation 10%