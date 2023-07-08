import argparse
import torch
from utils.configuration import CONFIG

from utils.training_utils import get_tokenizer, NER_MODEL
from utils.inference_utils import prediction_fn, show_names



def main():
    # runs end-to-end infernece for given text input
    # parse the args
    parser= argparse.ArgumentParser(description= "Inference for Bengali NER Task")
    parser.add_argument("--text", help= "Text for inference. (list of texts is preferable)",
                        default= "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name, \
                        help="Provide a valid huggingface language model for tokne classification", \
                        choices=["nafi-zaman/celloscope-28000-ner-banglabert-finetuned", 
                                "csebuetnlp/banglabert",
                                "nafi-zaman/mbert-finetuned-ner"])
    parser.add_argument("--model_checkpoint", type= str, default= CONFIG.model_checkpoint, help= "Path to the saved model file")
    
    
    args = parser.parse_args()
    text= args.text
    CONFIG.model_name= args.model_name
    CONFIG.model_checkpoint= args.model_checkpoint

    # loading model and weights
    tokenizer = get_tokenizer(model_name= CONFIG.model_name)
    model= NER_MODEL(cfg= CONFIG)
    model.load_state_dict(torch.load(CONFIG.model_checkpoint, map_location=  CONFIG.device))
    model.to(CONFIG.device)

    if text != None:
        ## run inference for given text input
        outputs=[]
        if type(text) == str:
            output= prediction_fn(text, model, tokenizer, CONFIG)
            outputs.append(output)
            # print(text, outputs)
        elif type(text)== list:
            for txt in text:
                output= prediction_fn(txt, model, tokenizer, CONFIG)
                outputs.append(output)
                # print(txt, output)        
        else:
            outputs= None
            print("Please give input in string format or list of strings")

        if outputs != None:
            result= show_names(text, outputs, tokenizer)
    else:
        print("Please give text input in proper format")


if __name__ == "__main__":
    main()