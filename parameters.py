

videoNames = [
"001.avi",
"002.avi",
"003.avi",
"004.avi",
"005.avi",
"006.avi",
"007.avi",
"008.avi",
"009.avi",
"010.avi",
"011.avi",
"012.avi",
"013.avi",
"014.avi",
"015.avi",
"016.avi",
"017.avi",
"018.avi",
"019.avi",
"020.avi",
"021.avi",
"022.avi",
"023.avi",
"024.avi",
"025.avi",
"026.avi",
"027.avi",
"028.avi",
"029.avi",
"030.avi",
"031.avi",
"032.avi",
"033.avi",
"034.avi",
"035.avi",
"036.avi",
"037.avi",
"038.avi",
"039.avi",
"040.avi",
"041.avi",
"042.avi",
"043.avi",
"044.avi",
"045.avi",
"046.avi",
"047.avi",
"048.avi",
"049.avi",
"050.avi",
"051.avi",
"052.avi",
"053.avi",
"054.avi",
"055.avi",
"056.avi",
"057.avi",
"058.avi",
"059.avi",
"060.avi",
"061.avi",
"062.avi",
"063.avi",
"064.avi",
"065.avi",
"066.avi",
"067.avi",
"068.avi",
"069.avi",
"070.avi",
"071.avi",
"072.avi",
"073.avi",
"074.avi",
"075.avi",
"076.avi",
"077.avi",
"078.avi",
"079.avi",
"080.avi",
"081.avi",
"082.avi",
"083.avi",
"084.avi",
"085.avi",
"086.avi",
"087.avi",
"088.avi",
"089.avi",
"090.avi",
"091.avi",
"092.avi",
"093.avi",
"094.avi",
"095.avi",
"096.avi",
"097.avi",
"098.avi",
"099.avi",
"100.avi",
"101.avi",
"102.avi"
]
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
