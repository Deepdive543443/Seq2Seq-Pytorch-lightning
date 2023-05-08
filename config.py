

args = {
    'BATCH_SIZE' : 400,
    'LEARNING_RATE' : 1e-4,
    'EPOCHS' : 200,


    # Model configuraion
    'ATTENTION_HEAD' : 1,
    'ENCODER_LAYERS' : 1,
    'DECODER_LAYERS' : 1,# This needs to be two times of encoder if using bidirection
    'EMB_DIM' : 300,
    'BIDIRECTION' : False
}


