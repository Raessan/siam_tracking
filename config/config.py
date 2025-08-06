import numpy as np

# DATASET PARAMETERS
DIR_DATA = '/home/rafa/deep_learning/datasets/LaSOT' # Root to the folder with the prepared data
SIZE_TEMPLATE = 127
SIZE_SEARCH = 255
SIZE_OUT = 25
MAX_FRAME_SEP = 10
NEG_PROB = 0.3
EXTRA_CONTEXT_TEMPLATE = 0.25
MIN_EXTRA_CONTEXT_SEARCH = 0.25
MAX_EXTRA_CONTEXT_SEARCH = 0.75
MAX_SHIFT = 32
REG_FULL = False
IMG_AUGMENT_TRAINING = True
IMG_AUGMENT_VALID = False
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
# MODEL PARAMETERS
BATCH_SIZE = 32
LAYERS_FREEZE = 3 #["stem", "layer1", "layer2", "layer3"]

# TRAINING PARAMETERS
THRESHOLD_CLS = 0.5
ALPHA_LOSS = 0.25
GAMMA_LOSS = 2.0
WEIGHT_LOSS = 1.0

LEARNING_RATE = 0.0001
NUM_EPOCHS = 15
NUM_SAMPLES_PLOT = 6

LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH_TRAIN_LOAD = '/home/rafa/deep_learning/projects/siam_tracking/results/2025-08-02_00-20-39/model_2.pth'
RESULTS_PATH = '/home/rafa/deep_learning/projects/siam_tracking/results'

# PARAMETERS FOR INFERENCE
MODEL_PATH_INFERENCE = '/home/rafa/deep_learning/projects/siam_tracking/results/2025-08-02_20-03-42/model_0.pth'
PIXEL_OFFSET_PER_FRAME = 5