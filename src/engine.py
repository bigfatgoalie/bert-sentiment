#for training and evaluation functions
from tqdm import tqdm
import torch
import torch.nn as nn

def loss_fn(outputs,targets):
    return nn.BCEWithLogitsLoss()(outputs,targets.view(-1,1))


def train_fn(data_loader, model, optimizer, device,scheduler):# accumulation_steps):
    model.train()
    
    for _, dataset  in tqdm(enumerate(data_loader), total=len(data_loader)):
        #extract stuff here
        ids = dataset["ids"]
        token_type_ids = dataset["token_type_ids"]
        mask = dataset["mask"]
        targets = dataset["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask = mask,
            token_type_ids = token_type_ids
        )

        loss = loss_fn(outputs,targets)
        loss.backward()

        #if (batch_index+1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, dataset  in tqdm(enumerate(data_loader), total=len(data_loader)):
            #extract stuff here
            ids = dataset["ids"]
            token_type_ids = dataset["token_type_ids"]
            mask = dataset["mask"]
            targets = dataset["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask = mask,
                token_type_ids = token_type_ids
            )
            #loss should be calculated in eval_fn for monitoring and accuracy purposes
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs,fin_targets



