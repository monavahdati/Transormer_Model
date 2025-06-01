# -*- coding: utf-8 -*-
"""
User Classification Model using Transformer Architecture

This script implements a user classification model utilizing a transformer architecture to predict user behavior and credit eligibility.
"""

# Install required packages
!pip install shap

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             precision_recall_curve, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import io

# Load the dataset
uploaded = files.upload()
print(uploaded.keys())
data = pd.read_csv(io.BytesIO(uploaded['bnpl_credit_data_final.csv']))

# Preprocess the data
data.fillna(0, inplace=True)
data['Age Condition'] = np.where(data['Age'] < 18, 0, 1)
data['Credit_Condition'] = np.where(data['Credit Score'] > 519, 1, 0)

# Define purchase columns
purchase_freq_cols = [f'Monthly Purchase Frequency {i}' for i in range(1, 7)]
purchase_amount_cols = [f'Monthly Purchase Amount {i}' for i in range(1, 7)]

data['Total_Purchase_Frequency'] = data[purchase_freq_cols].sum(axis=1)
data['Total_Purchase_Amount'] = data[purchase_amount_cols].sum(axis=1)
data['Repeat Usage'] = data['Repeat Usage'].map({'Yes': 1, 'No': 0})

# Function to determine credit amount and repayment period
def determine_credit(row):
    if row['Credit_Condition'] == 0:
        return 0, 0  # No credit
    if row['Payment Status'] == 'No':
        if row['Total_Purchase_Amount'] > 310000001:
            return 10000000, 1  # 10M for 1 month
        elif row['Total_Purchase_Amount'] > 150000001:
            return 5000000, 1  # 5M for 1 month
    else:
        if row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] > 220000000:
            return 10000000, 3  # 10M for 3 months
        elif row['Total_Purchase_Frequency'] > 79:
            return 10000000, 1  # 10M for 1 month
        elif row['Total_Purchase_Amount'] > 110000000:
            return 5000000, 3  # 5M for 3 months
        elif row['Total_Purchase_Amount'] < 110000001:
            return 5000000, 1  # 5M for 1 month
        elif row['Total_Purchase_Frequency'] < 41 and row['Total_Purchase_Amount'] < 80000001:
            return 2000000, 1  # 2M for 1 month
    return 0, 0  # Default no credit

data[['Credit Amount', 'Repayment Period']] = data.apply(determine_credit, axis=1, result_type='expand')

# Define target variable
data['Target'] = np.where(data['Credit_Condition'] & (data['Total_Purchase_Amount'] > 10), 1, 0)

# Prepare features and target
features = data[['Age', 'Credit Score', 'Total_Purchase_Frequency', 'Total_Purchase_Amount', 'Age Condition', 'Rating', 'Repeat Usage']]
target = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Model definition
class ImprovedTransformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, num_classes):
        super(ImprovedTransformer, self).__init__()
        self.input_layer = nn.Linear(features.shape[1], embed_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dim_feedforward=forward_expansion * embed_size, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc_out(x)

# Model parameters
embed_size = 128
heads = 8
num_layers = 4
forward_expansion = 4
dropout = 0.2
num_classes = 1

# Initialize the model
model = ImprovedTransformer(embed_size, heads, num_layers, forward_expansion, dropout, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.0006)
criterion = nn.BCEWithLogitsLoss()

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        predictions = torch.sigmoid(outputs).round()
        total_correct += (predictions == y_batch).sum().item()
        total_samples += y_batch.size(0)

    train_loss = total_loss / total_samples
    train_accuracy = total_correct / total_samples

    # Validation
    model.eval()
    with torch.no_grad():
        val_total_loss = 0
        val_total_correct = 0
        val_total_samples = 0
        all_predictions = []
        all_y_test = []

        for X_val_batch, y_val_batch in test_loader:
            val_outputs = model(X_val_batch)
            val_loss = criterion(val_outputs, y_val_batch)
            val_total_loss += val_loss.item() * X_val_batch.size(0)

            val_predictions = torch.sigmoid(val_outputs).round()
            val_total_correct += (val_predictions == y_val_batch).sum().item()
            val_total_samples += y_val_batch.size(0)

            all_predictions.extend(val_predictions.numpy())
            all_y_test.extend(y_val_batch.numpy())

    val_loss = val_total_loss / val_total_samples
    val_accuracy = val_total_correct / val_total_samples

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss:.4f} -- Val Loss: {val_loss:.4f} -- Train Acc: {train_accuracy:.4f} -- Val Acc: {val_accuracy:.4f}')

# Calculate final metrics
accuracy = accuracy_score(all_y_test, all_predictions)
recall = recall_score(all_y_test, all_predictions)
precision = precision_score(all_y_test, all_predictions)
f1 = f1_score(all_y_test, all_predictions)
conf_matrix = confusion_matrix(all_y_test, all_predictions)
auc_score = roc_auc_score(all_y_test, all_predictions)

print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}')
print('Confusion Matrix:\n', conf_matrix)

# Save the model
torch.save(model.state_dict(), 'improved_transformer_model.pth')

# Create DataFrame for predictions
results_df = pd.DataFrame({'Actual': all_y_test, 'Predicted': all_predictions.flatten()})
results_df.to_csv('customer_credit_offers_Transformer.csv', index=False)

# SHAP Explainer
explainer_shap = shap.Explainer(model_predict, X_train_tensor.numpy())
shap_values = explainer_shap(X_test_tensor.numpy())
feature_names = features.columns.tolist()
shap.summary_plot(shap_values, X_test_tensor.numpy(), feature_names=feature_names)

print("Results saved to 'customer_credit_offers_Transformer.csv'")
