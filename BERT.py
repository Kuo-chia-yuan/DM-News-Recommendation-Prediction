# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 載入資料集
train_behaviors = pd.read_csv('train_behaviors.tsv', delimiter='\t')
train_news = pd.read_csv('train_news.tsv', delimiter='\t')
test_behaviors = pd.read_csv('test_behaviors.tsv', delimiter='\t')
test_news = pd.read_csv('test_news.tsv', delimiter='\t')

train_behaviors = train_behaviors.head(int(len(train_behaviors) * 1))
train_news = train_news.head(int(len(train_news) * 1))
test_behaviors = test_behaviors.head(int(len(test_behaviors) * 1))
test_news = test_news.head(int(len(test_news) * 1))

print("Train Behaviors:")
print(train_behaviors.head())

print("\nTrain News:")
print(train_news.head())

print("\nTest Behaviors:")
print(test_behaviors.head())

print("\nTest News:")
print(test_news.head())

# 建立BERT模型及其tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

from tqdm import tqdm 

# 合併訓練和測試資料的新聞標題和摘要
def merge_news(df_behaviors, df_news):
    # 將clicked_news和impressions列的字串轉換為列表，先檢查是否為有效的字串
    df_behaviors['clicked_news'] = df_behaviors['clicked_news'].apply(lambda x: x.split() if isinstance(x, str) else [])
    df_behaviors['impressions'] = df_behaviors['impressions'].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    merged_df = pd.DataFrame(columns=['user_id', 'news_id', 'title', 'abstract'])
    
    # 使用tqdm來顯示進度條
    for index, row in tqdm(df_behaviors.iterrows(), total=len(df_behaviors), desc="Processing rows"):
        clicked_news = row['clicked_news']
        
        # 迭代點擊的每個新聞
        for news_id in clicked_news:
            if news_id.startswith('N'):
                news_info = df_news[df_news['news_id'] == news_id]
                if not news_info.empty:
                    news_title = news_info['title'].values[0]
                    news_abstract = news_info['abstract'].values[0]
                    merged_df = pd.concat([merged_df, pd.DataFrame({
                        'user_id': [row['user_id']],
                        'news_id': [news_id],
                        'title': [news_title],
                        'abstract': [news_abstract]
                    })], ignore_index=True)
    
    return merged_df

# 印出訓練資料的進度
train_data = merge_news(train_behaviors, train_news)
print("Train Data:")
print(train_data.head())

# 印出測試資料的進度
test_data = merge_news(test_behaviors, test_news)
print("\nTest Data:")
print(test_data.head())

MAX_LEN = 128
def tokenize_data(tokenizer, df):
    input_ids = []
    attention_masks = []

    for index, row in df.iterrows():
        # Handle NaN in title or abstract
        title = str(row['title']) if not pd.isnull(row['title']) else ""
        abstract = str(row['abstract']) if not pd.isnull(row['abstract']) else ""

        # Tokenize title and abstract
        encoded_dict = tokenizer.encode_plus(
                            title,
                            abstract,
                            add_special_tokens = True,
                            max_length = MAX_LEN,
                            padding = 'max_length',
                            truncation = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

# 定義模型架構
class NewsClickModel(nn.Module):
    def __init__(self, bert_model, num_labels=15):
        super(NewsClickModel, self).__init__()
        self.bert = bert_model
        self.cls_layer = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 使用BERT的CLS token作為輸出
        logits = self.cls_layer(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities

# 初始化模型、損失函數和優化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NewsClickModel(bert_model).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

from tqdm import tqdm  # 引入 tqdm

def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader))  # 初始化進度條
        
        for i, (inputs, masks, labels) in progress_bar:
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")  # 更新進度條顯示
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss / len(train_loader):.4f}")

import numpy as np

from tqdm import tqdm  # 導入 tqdm

def predict_model(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        # 使用 tqdm 來包裝 test_loader，並顯示進度條
        for inputs, masks in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            outputs = model(inputs, masks)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions

# 將預測結果格式化成CSV輸出
def format_predictions(predictions, test_data):
    result_df = pd.DataFrame(columns=['id'] + [f'p{i+1}' for i in range(predictions.shape[1])])
    result_df['id'] = test_data['user_id']
    result_df.iloc[:, 1:] = predictions
    return result_df

# 主程式流程
if __name__ == "__main__":
    # 處理訓練資料
    train_input_ids, train_attention_masks = tokenize_data(tokenizer, train_data)
    train_labels = torch.zeros(train_input_ids.shape[0], 15)  # 暫時假設標籤為0，實際上需根據點擊情況修改
    
    # 設定訓練資料的DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 訓練模型
    train_model(model, train_loader, criterion, optimizer)
    
    # 處理測試資料
    test_input_ids, test_attention_masks = tokenize_data(tokenizer, test_data)
    
    # 設定測試資料的DataLoader
    test_dataset = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 進行預測
    predictions = predict_model(model, test_loader)
    
    # 格式化預測結果並輸出成CSV檔案
    result_df = format_predictions(predictions, test_data)
    result_df.to_csv('predictions.csv', index=False)
