import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item): #takes an item and returns ids, token_type_ids etc.
        review = str(self.review[item]) #not needed, just for sanity
        review = " ".join(review.split()) #to remove spaces

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens = True, #to add special tokens such as cls, sep etc.
            max_length = self.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # padding_length = self.max_len - len(ids)
        # #for BERT padding needed on right side
        # ids = ids + ([0]*padding_length)
        # mask = mask + ([0]*padding_length)
        # token_type_ids = token_type_ids + ([0]*padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float) #dtype of target depends on number of outputs and optimizer function used
                                                                         #cross-entropy use torch.long
        }





