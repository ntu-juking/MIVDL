import pandas as pd
import torch
from transformers import ImageGPTForImageClassification, BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import  tqdm_notebook
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
neg_data = pd.read_csv('../raw_data/test.csv')
train_df, test_df = train_test_split(neg_data, test_size=0.2, random_state=42)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(texts, labels):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    labels = labels.to_list()
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels
train_input_ids, train_attention_masks, train_labels = preprocess_data(train_df['content'], train_df['label'])
test_input_ids, test_attention_masks, test_labels = preprocess_data(test_df['content'], test_df['label'])


train_data = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_data = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

lr = 1e-5
max_grad_norm = 1.0
weight_decay = 1e-2
num_total_steps = len(train_dataloader)
no_decay = ['bias' , 'LayerNorm.weight']
optimizer_grouped_parameters = [
{'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
{'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0}]

### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)  # To reproduce BertAdam specific behavior set correct_bias=False
 # PyTorch scheduler

total_step = len(train_dataloader)

# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 10

# trange is a tqdm wrapper around the normal python range
for epoch in tqdm_notebook(range(epochs)):

    # Training
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    total_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    i = 0
    # Train the data for one epoch
    for batch in train_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Forward pass
        optimizer.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # scheduler.step()

    print(f'The epoch is {epoch}, the total_loss is {total_loss}, average_loss is {total_loss / total_step}')

    model.eval()
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(test_dataloader):
          batch = tuple(t.to(device) for t in batch)
          # Unpack the inputs from our dataloader
          b_input_ids, b_input_mask, b_labels = batch
          # Forward pass
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          # print (outputs)
          prediction = torch.argmax(outputs[0],dim=1)
          total += b_labels.size(0)
          correct+=(prediction==b_labels).sum().item()
    print(f'The {epoch} test--------------------')
    print('Test Accuracy of the model on val data is: {} %'.format(100 * correct / total))