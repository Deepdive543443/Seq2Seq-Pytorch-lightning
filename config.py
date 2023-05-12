args = {
    # Training
    'DEVICE': 'cuda',
    'BATCH_SIZE' : 256,
    'LEARNING_RATE' : 1e-4,
    'EPOCHS' : 300,
    #'TEACH_RATE': 0.5,


    # Model configuraion
    'ENCODER_TYPE' : 'GRU', # 'GRU'
    'DECODER_TYPE' : 'GRU',
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 1,
    'DECODER_LAYERS' : 2,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 300,
    'HIDDEN_ENCODER': 1024,
    'HIDDEN_DECODER': 1024,
    'DROPOUT_ENCODER': 0.5,
    'DROPOUT_DECODER': 0.5,
    'BIDIRECTION' : True,

    #Dataset
    'PAIR':('de', 'en')
}