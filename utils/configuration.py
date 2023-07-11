import torch

class CONFIG:

    train= True #False #True
    debug= True #False #
    seed= 20
    output_dir= "./Models/"
    data_path1_train= "./Datasets/dataset_1_train.txt"
    test_data_path= "./Datasets/dataset_1_test.txt"
    data_path2= "./Datasets/dataset_2.jsonl"

    n_folds= 3
    num_epochs= 10
    label_names=['O', 'B-PER', 'I-PER']
    num_labels= len(label_names)
    model_name= "nafi-zaman/celloscope-28000-ner-banglabert-finetuned"  ##"csebuetnlp/banglabert" #"csebuetnlp/banglabert_large"  #"nafi-zaman/mbert-finetuned-ner" #
    model_checkpoint= "./Models/best_model_0.bin"
    max_length= 126

    dataset_no= 3 # 1: Train on dataset-1 # 2: Train on dataset-2 # 3: Train on combined dataset
    do_normalize= True
    do_downsampling= True
    do_upsampling= True
    upsample_size= 0.5
    
    train_batch_size= 8
    valid_batch_size= 16
    test_batch_size= 16
    num_workers= 2
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    patience= 3
    gradient_accumulation_steps= 1
    learning_rate= 2e-5 #5e-5
    weight_decay= 1e-1
    scheduler= "linear"
    T_max= 500
    T_0= 500
    min_lr= 1e-7
    eps = 1e-6
    betas= [0.9, 0.999]

    if debug:
        n_folds= 2
        num_epochs=2
        dataset_size= 300
