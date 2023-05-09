

args = {
    'BATCH_SIZE' : 64,
    'LEARNING_RATE' : 1e-4,
    'EPOCHS' : 100,


    # Model configuraion
    'ENCODER_TYPE' : 'GRU', # 'GRU'
    'DECODER_TYPE' : 'GRU',
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 1,
    'DECODER_LAYERS' : 1,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 300,
    'BIDIRECTION' : False
}


