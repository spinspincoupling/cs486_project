
d_k = 64
d_v = 64
d_ff = 2048
d_embedding = 512
num_head = 8
num_stacks = 4
encoder = 'resnet34'
num_frames = 103

LEARNING_RATE = 0.0001
TRAIN_STEPS = 1000
BATCH_SIZE = 1

SEQ_SIZE = 103
STRIDE = 2*SEQ_SIZE
IN_DEPTH = 2*SEQ_SIZE
IN_HEIGHT = 240
IN_WIDTH = 320
IN_CHANNEL = 3
