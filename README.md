# Sentiment Analysis with RoBERTa (93% Accuracy)

## Introduction

- **Presenter:** Yash Jagwani
- **Location:** India
- **Project Overview:**
  - Developed a sentiment analysis model using the IMDB movie reviews dataset.
  - Focused on data cleaning, exploratory data analysis (EDA), and leveraging RoBERTa for sentiment classification.
  - Discussed the redundancy of traditional NLP preprocessing methods with advanced models.

---

## Dataset Overview

- **Dataset:** IMDB movie reviews
- **Size:** 50,000 reviews labeled as positive or negative
- **Objective:** Accurately predict the sentiment of reviews

### Code Snippet: Data Loading

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Display the first 10 rows
df.head(10)

# Display basic information
df.info()
```

---

## Data Cleaning and EDA

- **Importance of Data Cleaning:**
  - Ensures accuracy and relevance for analysis and modeling.

### Steps for Data Cleaning:

1. **Convert to Lowercase**

   ```python
   df['review'] = df['review'].str.lower()
   ```

2. **Remove URLs and HTML Tags**

   ```python
   import re
   df['review'] = df['review'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))
   df['review'] = df['review'].apply(lambda x: re.sub(r'<.*?>', '', x))
   ```

3. **Remove Special Characters**

   ```python
   df['review'] = df['review'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]+', '', x))
   ```

4. **Expand Slang Terms**

   ```python
   slang_dict = {
       # ... (slang terms and their expansions)
   }
   df['review'] = df['review'].apply(lambda x: " ".join([slang_dict.get(word, word) for word in x.split()]))
   ```

5. **Remove Stop Words**

   ```python
   from nltk.corpus import stopwords
   stop_words = set(stopwords.words('english'))

   def remove_stopwords(text):
       return " ".join([word for word in text.split() if word not in stop_words])

   df['review'] = df['review'].apply(remove_stopwords)
   ```

---

## Validation of Data Cleaning

- **Check for Remaining Issues:**
  - HTML tags
  - URLs
  - Slang terms
  - Special characters

### Code Snippet: Validation Functions

```python
def contains_html_tags(text):
    import re
    pattern = re.compile(r'<.*?>')
    return bool(pattern.search(text))

def contains_url(text):
    import re
    pattern = re.compile(r'http\S+|www.\S+')
    return bool(pattern.search(text))

def contains_slang(text):
    slang_terms = set(slang_dict.keys())
    return any(word in slang_terms for word in text.split())

def contains_special_characters(text):
    import re
    pattern = re.compile(r'[^A-Za-z0-9\s]+')
    return bool(pattern.search(text))

def contains_emoji(text):
    import emoji
    return emoji.demojize(text) != text
```

---

## Exploratory Data Analysis (EDA)

- **Check for Missing Values and Duplicates**

### Code Snippet: Missing Values and Duplicates

```python
# Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values in each column:\n{missing_values}")

# Identify duplicates
duplicates = df[df.duplicated(subset='review')]
print(f"Number of duplicate reviews: {duplicates.shape[0]}")

# Remove duplicates
df = df.drop_duplicates(subset='review').reset_index(drop=True)
```

---

### Sentiment Distribution

- **Visualization:**

  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.countplot(x='sentiment', data=df)
  plt.title('Distribution of Sentiment Classes')
  plt.xlabel('Sentiment')
  plt.ylabel('Count')
  plt.show()
  ```

- **Observation:** Balanced dataset with equal positive and negative reviews.

---

### Review Length Analysis

- **Distribution of Review Lengths:**

  ```python
  df['review_length'] = df['review'].apply(lambda x: len(x.split()))
  sns.histplot(df['review_length'], bins=50, kde=True)
  plt.title('Review Length Distribution')
  plt.xlabel('Number of Words')
  plt.ylabel('Frequency')
  plt.show()
  ```

- **Observation:** Average review length around 135 words.

---

### Common Words Analysis

- **Most Common Words in Reviews:**

  ```python
  from collections import Counter

  word_counts = Counter(" ".join(df['review']).split())
  common_words = word_counts.most_common(20)
  print(f"Most common words: {common_words}")
  ```

- **Observation:** Neutral words dominate; context matters for sentiment differentiation.

---

## Minimal Preprocessing for RoBERTa

- **Focus on Essential Cleaning Steps:**
  - Remove duplicates, URLs, and HTML tags.

### Code Snippet: Minimal Preprocessing

```python
# Reset the dataset and perform minimal cleaning
df = pd.read_csv('IMDB Dataset.csv')

# Remove duplicates
df = df.drop_duplicates(subset='review').reset_index(drop=True)

# Remove URLs and HTML tags
import re
df['review'] = df['review'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))
df['review'] = df['review'].apply(lambda x: re.sub(r'<.*?>', '', x))
```

---

## Redundancy of Traditional Preprocessing Methods

- **Why Skip Traditional Methods?**
  - **Stop Word Removal**
  - **Lemmatization and Stemming**
  - **Part-of-Speech (POS) Tagging**
  - **Bag of Words and TF-IDF**
  - **N-grams Feature Extraction**

### Observations:

- RoBERTa captures context and semantics effectively, making extensive preprocessing unnecessary.

---

## Tokenization and Model Architecture

- **RoBERTa Tokenizer:**
  - Utilizes Byte-Pair Encoding (BPE).

### Code Snippet: Tokenization

```python
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
```

---

## Model Training

### Code Snippet: Model Training

```python
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

# Load the pre-trained RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=1000,
    logging_dir='./logs',
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Assuming train_dataset is prepared
    eval_dataset=eval_dataset     # Assuming eval_dataset is prepared
)

# Start training
trainer.train()
```

---

## Model Evaluation

### Code Snippet: Model Evaluation

```python
# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
```

- **Observation:** 93% Accuracy.

---

## Key Takeaways

1. **Conducted thorough data cleaning and EDA.**
2. **Recognized redundancy of traditional preprocessing with modern models.**
3. **Leveraged RoBERTa’s capabilities for efficient sentiment analysis.**
4. **Achieved high accuracy with minimal preprocessing.**

---

## Conclusion

- **Final Thoughts:**
  - Embracing modern NLP models enhances robustness and efficiency.

---

## Thank You!

- **Call to Action:**
  - Looking forward to positive feedback!
  - Happy coding!

---

# Contact

**Yash Jagwani** - [Email](mailto:toyash58@gmail.com) - [GitHub](https://github.com/yummychill-dev)

---
