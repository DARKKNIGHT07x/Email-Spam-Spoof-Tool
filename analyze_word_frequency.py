import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from bs4 import BeautifulSoup  # Import BeautifulSoup for HTML parsing

# Load the dataset
data = pd.read_csv('Training_Dataset.csv')  # Replace with your dataset filename

# Strip any leading/trailing whitespaces in the columns
data['Category'] = data['Category'].str.strip()
data['Message'] = data['Message'].str.strip()

# Function to clean HTML content using BeautifulSoup
def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()  # Extract text from HTML

# Apply cleaning to the 'Message' column
data['Message'] = data['Message'].apply(clean_html)

# Separate HAM and SPAM messages
ham_messages = data[data['Category'] == 'ham']['Message']
spam_messages = data[data['Category'] == 'spam']['Message']

# Vectorize the messages using n-grams (unigrams and bigrams) and stopword removal
vectorizer = CountVectorizer(
    stop_words='english',   # Removes common words like "the," "is," etc.
    ngram_range=(1, 2),     # Uses both unigrams and bigrams (word pairs)
    min_df=2,               # Ignores words that appear in fewer than 2 documents
    max_df=0.8              # Ignores words that appear in more than 80% of documents
)

# Vectorize HAM and SPAM messages
ham_word_counts = vectorizer.fit_transform(ham_messages)
ham_vocab = vectorizer.get_feature_names_out()
ham_counts = ham_word_counts.sum(axis=0).A1

spam_word_counts = vectorizer.fit_transform(spam_messages)
spam_vocab = vectorizer.get_feature_names_out()
spam_counts = spam_word_counts.sum(axis=0).A1

# Combine words and their frequencies into dictionaries
ham_word_freq = dict(zip(ham_vocab, ham_counts))
spam_word_freq = dict(zip(spam_vocab, spam_counts))

# Get the common words between HAM and SPAM
common_words = set(ham_word_freq.keys()) & set(spam_word_freq.keys())

# Eliminate common words from one category based on frequency comparison
for word in common_words:
    if ham_word_freq[word] > spam_word_freq[word]:
        # Remove the word from SPAM
        del spam_word_freq[word]
    elif spam_word_freq[word] > ham_word_freq[word]:
        # Remove the word from HAM
        del ham_word_freq[word]

# Get the top 20 most frequent words for each category after elimination
ham_top_words = Counter(ham_word_freq).most_common(20)
spam_top_words = Counter(spam_word_freq).most_common(20)

# Print the results
print("Top 20 Words in HAM Emails (After Elimination):")
for word, count in ham_top_words:
    print(f"{word}: {count}")

print("\nTop 20 Words in SPAM Emails (After Elimination):")
for word, count in spam_top_words:
    print(f"{word}: {count}")

# Save results to CSV for better visualization
pd.DataFrame(ham_top_words, columns=['Word', 'Frequency']).to_csv('ham_top_words_after_elimination.csv', index=False)
pd.DataFrame(spam_top_words, columns=['Word', 'Frequency']).to_csv('spam_top_words_after_elimination.csv', index=False)
