from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import numpy as np

def bert_embeddings(X, type='sentence'):
    model_class, tokenizer_class, pretrained_weights = (
        TFBertModel,BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    tokenized = X.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    last_hidden_states = model(tokenized)[0]
    if type is 'words':
        return last_hidden_states[:, 1:, :].numpy()
    elif type is 'sentence':
        return last_hidden_states[:,0,:].numpy()
    else:
        return last_hidden_states


if __name__ == '__main__':
    pass