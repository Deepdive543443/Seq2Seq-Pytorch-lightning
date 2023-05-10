import json

args = {
    'DEVICE': 'cuda',


    'BATCH_SIZE' : 128,
    'LEARNING_RATE' : 1e-3,
    'EPOCHS' : 50,


    # Model configuraion
    'ENCODER_TYPE' : 'GRU', # 'GRU'
    'DECODER_TYPE' : 'GRU',
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 1,
    'DECODER_LAYERS' : 1,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 150,
    'HIDDEN_ENCODER': 75,
    'HIDDEN_DECODER': 75,
    'DROPOUT_ENCODER': 0.5,
    'DROPOUT_DECODER': 0.5,
    'BIDIRECTION' : False
}