import json

args = {
    'BATCH_SIZE' : 400,
    'LEARNING_RATE' : 1e-4,
    'EPOCHS' : 100,


    # Model configuraion
    'ENCODER_TYPE' : 'GRU', # 'GRU'
    'DECODER_TYPE' : 'GRU',
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 1,
    'DECODER_LAYERS' : 1,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 100,
    'HIDDEN_ENCODER': 32,
    'HIDDEN_DECODER': 32,
    'DROPOUT_ENCODER': 0.2,
    'DROPOUT_DECODER': 0.2,
    'BIDIRECTION' : False
}