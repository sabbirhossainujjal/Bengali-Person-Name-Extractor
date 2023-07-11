"""This scripts runs an end-to-end inference. It takse a text input and returns extracted names in the text.
"""

import argparse
import torch
from utils.configuration import CONFIG
from normalizer import normalize

from utils.training_utils import get_tokenizer, NER_MODEL
from utils.inference_utils import prediction_fn, show_names



def main():
    # runs end-to-end infernece for given text input
    # parse the args
    parser= argparse.ArgumentParser(description= "Inference for Bengali NER Task")
    parser.add_argument("--text", help= "Text for inference. (list of texts is preferable)",
                        default= "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name, \
                        help="Provide a valid huggingface language model name for token classification", \
                        choices=["nafi-zaman/celloscope-28000-ner-banglabert-finetuned", 
                                "csebuetnlp/banglabert",
                                "csebuetnlp/banglabert_large",
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
            text= normalize(text)
            output= prediction_fn(text, model, tokenizer, CONFIG)
            outputs.append(output)
        elif type(text)== list:
            for txt in text:
                txt= normalize(txt)
                output= prediction_fn(txt, model, tokenizer, CONFIG)
                outputs.append(output)
        else:
            outputs= None
            print("Please give input in string format or list of strings")

        if outputs != None:
            result= show_names(text, outputs, tokenizer)
    else:
        print("Please give text input in proper format")


if __name__ == "__main__":
    main()
"""
Please give text input as one of the following format.
text=  "আন্তর্জাতিক অপরাধ ট্রাইব্যুনাল-১-এর চেয়ারম্যান বিচারপতি এ টি এম ফজলে কবীর অবসর নিয়েছেন ।"
or
text= ["আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম",
       "আন্তর্জাতিক অপরাধ ট্রাইব্যুনাল-১-এর চেয়ারম্যান বিচারপতি এ টি এম ফজলে কবীর অবসর নিয়েছেন ।", 
       "ব্যাংকের চেয়ারম্যান ও ঢাকা বিশ্ববিদ্যালয়ের ইন্টারন্যাশনাল বিজনেস বিভাগের অধ্যাপক খন্দকার বজলুল হক প্রথম আলো ডটকমকে জানান, বিকেল তিনটা ৫০ মিনিটে তিনি এ ধরনের অভিযোগ পেয়েছেন।",
       "একই সঙ্গে অভিযোগ তদন্তে বিশ্ববিদ্যালয়ের কোষাধ্যক্ষ সায়েন উদ্দিনকে প্রধান করে পাঁচ সদস্যের কমিটি গঠন করা হয়েছে।",
      ]
"""
