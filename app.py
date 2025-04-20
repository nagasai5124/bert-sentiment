
import streamlit as st
import torch
import numpy as np
import transformers
import torch.nn as nn
device='cpu'

def __getitem__(review):
    review=' '.join(review.split())

    inputs=tokenizer.encode_plus(review,
                                      None,
                                      add_special_tokens=True,
                                      max_length=max_len,
                                      padding='max_length',
                                      truncation=True)
    ids=inputs.get('input_ids', inputs.get('inputs_ids'))
    mask=inputs['attention_mask']
    token_type_ids=inputs['token_type_ids']
    # Move tensors to the device (GPU)
    return {
        'ids':torch.tensor(ids,dtype=torch.long).to(device), # Move to device
        'mask':torch.tensor(mask,dtype=torch.long).to(device).unsqueeze(0), # Move to device and add a dimension
        'token_type_ids':torch.tensor(token_type_ids,dtype=torch.long).to(device) # Move to device
        }


class BERTBaseUncased(nn.Module):
  def __init__(self):
    super(BERTBaseUncased,self).__init__()
    self.bert=transformers.BertModel.from_pretrained(bert_path)
    self.bert_drop=nn.Dropout(0.3)
    self.out=nn.Linear(768,1)
  def forward(self,ids,mask,token_type_ids):
    o2=self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids).last_hidden_state[:,0,:]
    bo=self.bert_drop(o2)
    output=self.out(bo)
    return output


st.title("sentiment analysis using bert")
review=st.text_input("review")
submit=st.button("submit")

bert_path='bert-base-uncased'
model=torch.load("model.pth", weights_only=False,map_location='cpu')
tokenizer=transformers.BertTokenizer.from_pretrained(bert_path,do_lower_case=True)
model.to(device)
max_len=64

if submit:
  token_str=__getitem__(review)
  outputs_1=model(ids=token_str['ids'].unsqueeze(0),mask=token_str['mask'],token_type_ids=token_str['token_type_ids'])
  outputs = np.array(outputs_1.cpu().detach().numpy()) >= 0.5
  if outputs[0][0]==True:
    st.write("positive")
  else:
    st.write("negative")
