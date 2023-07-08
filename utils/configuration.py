import torch

class CONFIG:
    train= True #False #True
    debug= True #False #
    seed= 20
    output_dir= "Models/"
    data_path1_train= "/media/sabbir/E/Research/Bengali-NER/Dataset_1/Data/Input/train_data.txt"
    data_path1_test= "/media/sabbir/E/Research/Bengali-NER/Dataset_1/Data/Input/test_data.txt"
    data_path2= "/media/sabbir/E/Research/Bengali-NER/dataset_2.jsonl"

    n_folds= 3
    num_epochs= 100
    label_names=['O', 'B-PER', 'I-PER']
    num_labels= len(label_names)
    model_name= "nafi-zaman/celloscope-28000-ner-banglabert-finetuned" #"csebuetnlp/banglabert" #"nafi-zaman/mbert-finetuned-ner"#
    max_length= 500
    train_batch_size= 8
    valid_batch_size= 16
    num_workers= 2
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    gradient_accumulation_steps= 1
    learning_rate= 5e-5
    weight_decay= 1e-2
    scheduler= "CosineAnnealingWarmRestarts"
    T_max= 500
    T_0= 500
    min_lr= 1e-7
    
    eps = 1e-6
    betas= [0.9, 0.999]

    if debug:
        n_folds= 2
        num_epochs=2
        dataset_size= 300

    