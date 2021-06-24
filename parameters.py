
d_k = 64
d_v = 64
d_ff = 2048
d_model = 512
num_head = 8
num_stacks = 4
encoder = 'resnet34'

LEARNING_RATE = 0.0001
TRAIN_STEPS = 1000
BATCH_SIZE = 1

SEQ_SIZE = 16
STRIDE = 2*SEQ_SIZE
IN_DEPTH = 2*SEQ_SIZE
IN_HEIGHT = 224
IN_WIDTH = 224
IN_CHANNEL = 3
