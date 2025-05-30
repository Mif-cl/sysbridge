class Config:
    # torch parameters
    float32_matmul_precision = "high"
    seed = 42

    # dataset parameters
    train_dataset_path = "data/Im"
    val_test_size = 0.2

    # model parameters
    tokenizer_name = "gpt2"
    feature_extractor = "microsoft/swin-base-patch4-window7-224-in22k"
    encoder_name = "microsoft/swin-base-patch4-window7-224-in22k"
    decoder_name = "gpt2"
    max_length = 512
    
    # training parameters
    num_epochs = 60
    batch_size_train = 8
    batch_size_val = 8
    learning_rate = 1e-4
    warmup_steps = 400
    max_grad_norm = 1.0
    betas = (0.95, 0.98) 
    eps=1e-08

    # image parameters
    image_size=(224, 224)
    
    # checkpoint parameters
    checkpoint_dir = "checkpoints"
    eval_steps = 200

    # metric parameters
    bleu = "google_bleu"