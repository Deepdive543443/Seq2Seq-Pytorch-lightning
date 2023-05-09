import json

args = {
    'BATCH_SIZE' : 128,
    'LEARNING_RATE' : 3e-4,
    'EPOCHS' : 600,


    # Model configuraion
    'ENCODER_TYPE' : 'GRU', # 'GRU'
    'DECODER_TYPE' : 'GRU',
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 1,
    'DECODER_LAYERS' : 1,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 150,
    'HIDDEN_ENCODER': 75,
    'HIDDEN_DECODER': 75,
    'DROPOUT_ENCODER': 0.2,
    'DROPOUT_DECODER': 0.2,
    'BIDIRECTION' : False
}