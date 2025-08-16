# DATASET PARAMETERS
DIR_DATA = '/home/rafa/deep_learning/datasets/LaSOT' # Root to the folder with the prepared data
SIZE_TEMPLATE = 127 # Size of the template image
SIZE_SEARCH = 255 # Size of the search image
SIZE_OUT = 25 # Size of the labels (heatmap classifier and bbox regressor)
MAX_FRAME_SEP = 10 # Max number of frames to jump inside a video to get a positive image
NEG_PROB = 0.3 # Percentage of negative samples
EXTRA_CONTEXT_TEMPLATE = 0.25 # Extra content for the template image
MIN_EXTRA_CONTEXT_SEARCH = 0.25 # Minimum value of extra content for the search image
MAX_EXTRA_CONTEXT_SEARCH = 0.75 # Maximum value of extra content for the search image
MAX_SHIFT = 32 # Maximum shift of the object of search image to prevent it from being always centered. This acts like augmetnation
REG_FULL = False # If False, only w,h are regressed. If true, also the offsets dx, dy are regressed
IMG_AUGMENT_TRAINING = True # Whether to perform photogrametric augmentation in training
IMG_AUGMENT_VALID = False # Whether to perform photogrametric augmentation in validation
IMG_MEAN = [0.485, 0.456, 0.406] # Mean of the image that the backbone (e.g. ResNet) expects
IMG_STD = [0.229, 0.224, 0.225] # Std of the image that the backbone (e.g. ResNet) expects
    
# MODEL PARAMETERS
DINOV3_DIR = "/home/rafa/deep_learning/projects/siam_tracking/dinov3"
DINO_MODEL = "dinov3_vits16plus"
PROJ_DIM = 256
BATCH_SIZE = 32 # Batch size
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "inov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

# TRAINING PARAMETERS
THRESHOLD_CLS = 0.5 # Threshold to consider that an object has been detected in the heatmap
ALPHA_LOSS = 0.25 # Alpha value for the loss fcn
GAMMA_LOSS = 2.0 # Gamma value for the loss fcn
WEIGHT_LOSS = 1.0 # Weight of the regressor loss (with respect to the classifier)

LEARNING_RATE = 0.00001 # Learning rate
NUM_EPOCHS = 15 # Number of epochs
NUM_SAMPLES_PLOT = 6 # Number of samples to plot during training or validation

LOAD_MODEL = False # Whether to load an existing model for training
SAVE_MODEL = True # Whether to save the result from the training
MODEL_PATH_TRAIN_LOAD = '/home/rafa/deep_learning/projects/siam_tracking/results/2025-08-02_00-20-39/model_2.pth' # Path of the model to load
RESULTS_PATH = '/home/rafa/deep_learning/projects/siam_tracking/results' # Folder where the result will be saved

# PARAMETERS FOR INFERENCE
MODEL_PATH_INFERENCE = '/home/rafa/deep_learning/projects/siam_tracking/results/2025-08-02_20-03-42/model_0.pth' # Path of the model to perform inference
PIXEL_OFFSET_PER_FRAME = 5 # Maximum number of pixels that the inference algorithm can move between samples