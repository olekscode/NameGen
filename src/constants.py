USE_CUDA = False

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3

MAX_LENGTH = 40
MAX_SEQ_LENGTH = MAX_LENGTH+2 # 2 for EOS_token and SOS_token

BATCH_SIZE = 100

TEST_SIZE = 0.2
VALID_SIZE = 0.16