# Configuration file for DistilBERT sentiment project

DATASET_CSV = "dataset.csv"
TRAIN_CSV = "tweet/train_stockemo.csv"
VAL_CSV = "tweet/val_stockemo.csv"
TEST_CSV = "tweet/test_stockemo.csv"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
NUM_LABELS = 2

OUTPUT_DIR = "./results"
LOG_DIR = "./logs"

BATCH_SIZE = 16
NUM_EPOCHS = 3.0
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
SEED = 42

TRAILING_DAYS = 4
BREAK_THRESHOLD = 0.8
TRADE_THRESHOLD = 1.0
