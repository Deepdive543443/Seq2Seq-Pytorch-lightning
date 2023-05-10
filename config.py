import json

args = {
    # Training
    'DEVICE': 'cuda',
    'BATCH_SIZE' : 128,
    'LEARNING_RATE' : 1e-3,
    'EPOCHS' : 50,


    # Model configuraion
    'ENCODER_TYPE' : 'GRU', # 'GRU'
    'DECODER_TYPE' : 'GRU',
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 2,
    'DECODER_LAYERS' : 2,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 300,
    'HIDDEN_ENCODER': 1024,
    'HIDDEN_DECODER': 1024,
    'DROPOUT_ENCODER': 0.5,
    'DROPOUT_DECODER': 0.5,
    'BIDIRECTION' : False,

    #Dataset
    'PAIR':('de', 'en'),
    # Log
    # 'LOGDIR' : 'tb_logs',
}