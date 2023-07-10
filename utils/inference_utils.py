from utils.configuration import CONFIG

def prediction_fn(text, model, tokenizer, cfg= CONFIG):
    """ Generate prediction for given text
    Args:
        text
        model 
        tokenizer 
        cfg
    Returns:
        model prediction logits
    """

    inputs= tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
    outputs= model(inputs['input_ids'].to(cfg.device), inputs['attention_mask'].to(cfg.device))
    outputs= outputs.detach().cpu().numpy().argmax(axis= -1)[0, 1:-1]
    
    return outputs


def extract_spans(prediction):
    """Extract spans where name occurs. (indices of B-PER, I-PER tokens)
    [1 2 0 0 0 0 0 1 2 0] --> [[0,1], [7, 8]]
    Args:
        prediction : single prediction from model.
    Returns:
        list of name spans.
    """
    span_indices = [i for i, v in enumerate(prediction) if v != 0 ]
    span_list= []
    span= []

    # get span indices
    for i in range(len(span_indices)):
        if i == 0 or span_indices[i] != span_indices[i-1]+1:
            if span:
                span_list.append(span)

            span= [span_indices[i]]

        else:
            span.append(span_indices[i])
    if span:
        span_list.append(span)
    
    return span_list

def extract_names(text, span_list, tokenizer):
    """Extract names from the given text and span_list. 
    This function takes the tokens that may be name extracted in span list and decode it to text format.
    Args:
        text: text from where name should be extracted
        span_list : name span list in predictions, output of the extract_spans functions
        tokenizer : 
    Returns:
        list of name in the given sentences.
    """
    name_list= []
    if len(span_list) > 0:
        for span in span_list:
            tokens= tokenizer(text)['input_ids'][1:-1][span[0]:span[-1]+1]
            name= tokenizer.decode(tokens)
            name_list.append(name)
        return name_list
    else:
        return None

def show_names(texts, predictions, tokenizer):

    if type(texts)== str:
        texts= [texts]
    result= []

    for text, pred in zip(texts, predictions):
        span_list= extract_spans(pred)
        name_list= extract_names(text, span_list, tokenizer)
        result.append((text, name_list))
        print(f"Given Text: {text} \nExtracted Names:{name_list}")
    
    return result

