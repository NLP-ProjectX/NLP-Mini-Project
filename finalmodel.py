import os
import re
import cv2
import torch
import pandas as pd
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from tqdm import tqdm
import torch.nn as nn
import random
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set random seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Step 1: Enhanced OCR Setup with Advanced Image Processing
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image):
    """Apply advanced preprocessing to improve OCR quality"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 15, 2)
    
    # Denoise image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Improve contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Apply morphological operations to make text clearer
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return processed

def extract_text_from_images(folder_path):
    """Extract text from receipt images with enhanced preprocessing"""
    data = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found")
        return pd.DataFrame(columns=["filename", "text", "raw_text"])
    
    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Resize image if too small
                if min(img.shape[0], img.shape[1]) < 500:
                    scale = 2.0
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
                # Apply preprocessing
                processed_img = preprocess_image(img)
                
                # Try different OCR configurations for optimal results
                # PSM modes: 4 (sparse text), 6 (block of text)
                text1 = pytesseract.image_to_string(processed_img, config='--psm 6 --oem 3')
                text2 = pytesseract.image_to_string(processed_img, config='--psm 4 --oem 3')
                
                # Choose the longer text as it likely contains more information
                if len(text1) > len(text2):
                    text = text1
                else:
                    text = text2
                
                # Clean the extracted text
                cleaned_text = clean_text(text.strip())
                
                # Add to dataset
                data.append({
                    "filename": file, 
                    "text": cleaned_text, 
                    "raw_text": text.strip()
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Processed {len(data)} receipt images")
    return pd.DataFrame(data)

def clean_text(text):
    """Advanced text cleaning and normalization for receipts"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove control characters and non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Normalize dates (convert different formats to consistent style)
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', r'\1/\2/\3', text)
    
    # Normalize price formats (make dollar signs and decimals consistent)
    text = re.sub(r'(\$|\€)\s*(\d+[,.]\d{2})', r'\1\2', text)
    text = re.sub(r'(\d+),(\d{2})', r'\1.\2', text)  # Convert comma to decimal point in prices
    
    # Remove special characters but preserve those important for receipts
    text = re.sub(r'[^\w\s\$\.,\/\-:]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 🔹 Step 2: Extract Text from Cropped Receipts
folder_path = r"C:\Users\nanda\OneDrive\Desktop\NLP-Mini-Project\Cropped_Receipts"
df = extract_text_from_images(folder_path)

# 🔹 Step 3: Enhanced Automated Labeling with Robust Rules
def classify_text(text, raw_text=None):
    """Improved classification with detailed pattern matching"""
    if not text:
        return "other"
    
    text = text.strip().lower()
    raw_text = raw_text.strip().lower() if raw_text else text
    
    # Rule 1: Total Amount Detection
    total_patterns = [
        r'(total|amount|balance|sum|due|bill|pay)[:\s]*\$?\s*\d+\.\d{2}',
        r'\$?\s*\d+\.\d{2}\s*(total|amount|balance|due)',
        r'total\s*\$?\s*\d+\.\d{2}',
        r'subtotal\s*\$?\s*\d+\.\d{2}'
    ]
    
    if any(re.search(pattern, text) for pattern in total_patterns):
        return "total_amount"

    # Rule 2: Date Detection
    date_patterns = [
        r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{2,4}\b",
        r"\b\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{2,4}\b",
        r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/\d{2,4}\b"
    ]
    
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns):
        return "date"

    # Rule 3: Restaurant Name Detection
    restaurant_keywords = [
        "mcdonald", "subway", "starbucks", "kfc", "pizza hut", "domino", "burger king",
        "wendy", "taco bell", "chipotle", "dunkin", "restaurant", "cafe", "diner", 
        "grill", "bistro", "steakhouse", "pizzeria", "bakery", "coffee shop", "bar & grill",
        "kitchen", "eatery", "house", "garden", "express", "tavern", "pub", "bbq"
    ]
    
    # Check for restaurant name patterns
    if any(kw in text for kw in restaurant_keywords):
        # More likely to be restaurant name if:
        if len(text.split()) <= 5:
            return "restaurant_name"
        
        # Check if any restaurant keyword is at the beginning
        if any(text.startswith(kw) for kw in restaurant_keywords):
            return "restaurant_name"
        
        # Check if there are capitalized words in raw text (typical for brand names)
        if raw_text and any(word.isupper() and len(word) > 2 for word in raw_text.split()):
            return "restaurant_name"

    # Rule 4: Food Items Detection
    item_patterns = [
        r"\b\d+\s*x\s*\w+",  # quantity x item format
        r"\b\w+[\w\s]*\s+\$?\d+\.\d{2}\b",  # item price format
        r"\d+\s+[a-z]+\s+\d+\.\d{2}"  # quantity item price format
    ]
    
    food_keywords = [
        "burger", "fries", "coffee", "tea", "sandwich", "pizza", "salad", "chicken", 
        "drink", "combo", "meal", "breakfast", "lunch", "dinner", "soda", "water", 
        "juice", "beer", "wine", "appetizer", "dessert", "cheese", "bacon", "egg"
    ]
    
    # Check for food item patterns
    if (any(re.search(pattern, text) for pattern in item_patterns) or 
        any(kw in text for kw in food_keywords)):
        # Confirm it contains a price
        if re.search(r'\$?\d+\.\d{2}', text):
            return "items"
    
    # Rule 5: Address detection
    address_keywords = [
        "street", "avenue", "road", "blvd", "boulevard", "drive", "lane", "st", "ave", 
        "rd", "dr", "highway", "hwy", "suite", "unit", "plaza", "mall", "center"
    ]
    
    if any(kw in text for kw in address_keywords) or re.search(r'\b\d{5}(-\d{4})?\b', text):
        return "address"
    
    # Default case
    return "other"

# Apply classification
df["category"] = df.apply(lambda row: classify_text(row["text"], row["raw_text"]), axis=1)

# Print category distribution
category_counts = df["category"].value_counts()
print(f"\nInitial category distribution:")
print(category_counts)

# Remove unclassified data
df_classified = df[df["category"] != "other"].reset_index(drop=True)

# Save labeled data
df_classified.to_csv("auto_labeled_receipt_data.csv", index=False)
print(f"✅ Auto-labeling complete! Data saved with {len(df_classified)} classified samples")

# 🔹 Step 4: Data Augmentation and Preparation
def augment_text(text, category):
    """Apply text augmentation techniques appropriate for receipt data"""
    augmented = []
    
    # Original text
    augmented.append(text)
    
    # Skip short texts
    if len(text) < 10:
        return augmented
    
    # Technique 1: Random character deletion (mild)
    chars = list(text)
    for _ in range(min(3, len(text) // 10)):
        if len(chars) > 3:
            pos = random.randint(0, len(chars) - 1)
            chars.pop(pos)
    augmented.append(''.join(chars))
    
    # Technique 2: Word swapping (for multi-word texts)
    words = text.split()
    if len(words) > 3:
        for _ in range(min(2, len(words) // 3)):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        augmented.append(' '.join(words))
    
    # Technique 3: Category-specific augmentations
    if category == "total_amount" and re.search(r'\$?\d+\.\d{2}', text):
        # Modify price slightly
        modified = re.sub(r'(\d+)\.(\d{2})', 
                          lambda m: f"{int(m.group(1)) + random.randint(-2, 2)}.{int(m.group(2)) + random.randint(-5, 5):02d}", 
                          text)
        augmented.append(modified)
    elif category == "date":
        # Modify date format
        if "/" in text:
            modified = text.replace("/", "-")
            augmented.append(modified)
    
    return augmented

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert categories to numerical labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df_classified["label"] = label_encoder.fit_transform(df_classified["category"])

# Map categories to integers
category_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
reverse_mapping = {v: k for k, v in category_mapping.items()}
print(f"\nCategory mapping: {category_mapping}")

# Apply data augmentation for balancing classes
class_counts = df_classified["category"].value_counts()
max_count = max(class_counts.values)
augmented_data = []

# Augment underrepresented classes
for category in class_counts.index:
    category_df = df_classified[df_classified["category"] == category]
    
    if len(category_df) < max_count:
        # Calculate how many samples to add
        samples_needed = max_count - len(category_df)
        
        # Sample from the category (with replacement if needed)
        samples_to_augment = category_df.sample(n=min(samples_needed, len(category_df) * 3), 
                                               replace=True)
        
        for _, row in samples_to_augment.iterrows():
            augmented_texts = augment_text(row["text"], row["category"])
            
            for aug_text in augmented_texts[1:]:  # Skip the original
                augmented_data.append({
                    "filename": f"{row['filename']}_aug",
                    "text": aug_text,
                    "raw_text": row["raw_text"],
                    "category": row["category"],
                    "label": row["label"]
                })

# Add augmented data to main dataframe
aug_df = pd.DataFrame(augmented_data)
df_balanced = pd.concat([df_classified, aug_df], ignore_index=True)

print(f"Dataset after augmentation: {len(df_balanced)} samples")
print(f"Balanced category distribution: {df_balanced['category'].value_counts()}")

# Stratified train-test-validation split
X = df_balanced["text"]
y = df_balanced["label"]

# Split into train+val and test sets (80/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split train into train and validation (80% of 80% = 64%, 20% of 80% = 16%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Custom Dataset class
class ReceiptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]) if hasattr(self.texts, 'iloc') else str(self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Create datasets with increased max_len for better context
train_dataset = ReceiptDataset(X_train, y_train, tokenizer, max_len=128)
val_dataset = ReceiptDataset(X_val, y_val, tokenizer, max_len=128)
test_dataset = ReceiptDataset(X_test, y_test, tokenizer, max_len=128)

# Optimized batch size based on available resources
batch_size = 16 if torch.cuda.is_available() else 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 🔹 Step 5: Improved BERT Model Architecture
class EnhancedBERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(EnhancedBERTClassifier, self).__init__()
        # Load pre-trained BERT with dropout regularization
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=num_classes,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
        # Freeze the first few layers to prevent overfitting (optional)
        # This helps when dataset is small
        # Uncomment to enable layer freezing:
        # for param in list(self.bert.parameters())[:30]:
        #     param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

# Initialize model
model = EnhancedBERTClassifier(num_classes=len(category_mapping), dropout_rate=0.3).to(device)

# 🔹 Step 6: Evaluation Function
def evaluate_model(model, dataloader, loss_fn=None):
    """Evaluate model performance on given dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if loss_fn:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Print detailed classification report
    if not loss_fn:  # Only when not calculating loss (final evaluation)
        print("\nClassification Report:")
        target_names = [reverse_mapping[i] for i in sorted(reverse_mapping.keys())]
        print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Return metrics
    if loss_fn:
        return total_loss / len(dataloader), accuracy, f1
    else:
        return accuracy, f1, precision, recall

# 🔹 Step 7: Advanced Training with Learning Rate Scheduling
def train_model(model, train_dataloader, val_dataloader, epochs=20):
    """Train model with learning rate scheduling and early stopping"""
    # Optimizer with weight decay for regularization
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Early stopping parameters
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        total_loss = 0
        
        # Progress bar for training batches
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Get batch data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Evaluate on validation set
        val_loss, val_acc, val_f1 = evaluate_model(model, val_dataloader, loss_fn)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Check early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print("✓ New best model saved!")
        else:
            patience_counter += 1
            print(f"⚠ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("\nLoaded best model from training")
    
    # Plot training history
    plot_training_history(history)
    
    # Save the best model to disk
    torch.save(model.state_dict(), "best_receipt_classifier.pt")
    print("Best model saved to best_receipt_classifier.pt")
    
    return model, history

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.plot(history['val_f1'], label='Val F1 Score')
    plt.title('Validation Metrics during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")

# 🔹 Step 8: Train the Model
print("\n🚀 Starting model training...")
model, history = train_model(model, train_dataloader, val_dataloader, epochs=20)

# 🔹 Step 9: Final Model Evaluation
print("\n📊 Final model evaluation on test set:")
test_acc, test_f1, test_precision, test_recall = evaluate_model(model, test_dataloader)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# 🔹 Step 10: Example Predictions
def predict_category(text, model=model, tokenizer=tokenizer):
    """Predict category for a given text"""
    # Preprocess the text
    cleaned_text = clean_text(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    # Get category name from prediction
    predicted_category = reverse_mapping[pred]
    
    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "predicted_category": predicted_category,
        "confidence": torch.softmax(outputs, dim=1).cpu().numpy()[0][pred]
    }

# Example predictions
print("\n🧪 Testing model on sample texts:")
sample_texts = [
    "$34.56 TOTAL",
    "05/12/2023",
    "McDonald's",
    "2 x Cheeseburger $8.99"
]

for text in sample_texts:
    result = predict_category(text)
    print(f"\nText: {result['text']}")
    print(f"Cleaned: {result['cleaned_text']}")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.4f}")

# 🔹 Step 11: Process a New Image (Optional)
def process_and_classify_receipt_image(image_path):
    """Process a receipt image and classify its text"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Extract text
        text = pytesseract.image_to_string(processed_img)
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Classify text
        result = predict_category(cleaned_text)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "predicted_category": result["predicted_category"],
            "confidence": result["confidence"]
        }
    
    except Exception as e:
        return {"error": str(e)}


new_image_path = r"C:\Users\nanda\OneDrive\Desktop\NLP-Mini-Project\Cropped_Receipts\cropped_receipt_29.png"
result = process_and_classify_receipt_image(new_image_path)
if "error" in result:
     print(f"Error: {result['error']}")
else:
     print(f"Original text: {result['original_text'][:100]}...")
     print(f"Cleaned text: {result['cleaned_text']}")
     print(f"Predicted category: {result['predicted_category']}")
     print(f"Confidence: {result['confidence']:.4f}")

print("\n✅ Receipt text classifier complete!")
print(f"Final accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")