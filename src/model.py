import transformers
import torch.nn as nn
import config

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased,self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,1)

    def forward(self, ids, mask, token_type_ids): #mask == attention mask
        #out1,_ == sequence of hidden states for each and every token for all batches (512 vectors of size 768)
        #        also called last hidden states
        #out2,o2 == only first hidden state, pooler output from the BERT pooler layer

        _, o2 = self.bert( 
            ids, 
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        #bert output
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

        




