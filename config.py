args = {
    # Training
    'DEVICE': 'cuda',
    'BATCH_SIZE' : 128,
    'LEARNING_RATE' : 1e-4,
    'EPOCHS' : 600,
    #'TEACH_RATE': 0.5,


    # Model configuraion
    'ENCODER_TYPE' : 'GRU', # 'GRU'
    'DECODER_TYPE' : 'GRU',
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 2,
    'DECODER_LAYERS' : 2,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 150,
    'HIDDEN_ENCODER': 512,
    'HIDDEN_DECODER': 512,
    'DROPOUT_ENCODER': 0.5,
    'DROPOUT_DECODER': 0.5,
    'BIDIRECTION' : False,

    #Dataset
    'PAIR':('en', 'de')
}