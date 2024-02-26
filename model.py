from transformers import (
    BertTokenizerm, 
    BertForSequenceClassification
)

from utils import hardmax

def get_tokenizer()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                            do_lower_case=True)
    return tokenizer

def get_model(weight_path="weights/best.model"):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        # No attention and hidden state as an output
                                                        output_attentions=False,
                                                        output_hidden_states=False)


def pipeline(text,
             tokenizer=tokenizer,
             model=model,
             device=device):

  tokenizer_configs =  dict(
      add_special_tokens=True,
      return_attention_mask=True,
      padding='longest',
      max_length=256,
      return_tensors='pt'
    )
  # Encode text into input_ids and attention_masks
  encoded_word = tokenizer.batch_encode_plus([text],**tokenizer_configs)
  input_ids = encoded_word['input_ids'].to(device)
  attention_masks = encoded_word['attention_mask'].to(device)
  model = model.to(device)
  # Inference
  outputs = model(input_ids=input_ids,
                  attention_mask=attention_masks)

  prob = softmax(outputs.logits,dim=1)
  prob = prob.detach().cpu().numpy()
  pred = hardmax(prob)
  return prob, pred