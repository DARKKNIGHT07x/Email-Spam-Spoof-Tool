import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import tkinter as tk
from tkinter import ttk, messagebox

# Function to load top words from CSV files (assuming the first code generated these CSVs)
def load_top_words():
    ham_top_words_df = pd.read_csv('ham_top_words_after_elimination.csv')
    spam_top_words_df = pd.read_csv('spam_top_words_after_elimination.csv')
    
    # Combine top words from ham and spam categories
    top_words = set(ham_top_words_df['Word']).union(set(spam_top_words_df['Word']))
    return list(top_words)

def train_spam_model(filename='Training_Dataset.csv'):
    """Train a model to detect spam emails using the updated dataset."""
    # Load dataset
    df = pd.read_csv(filename)
    
    # Ensure columns are used correctly
    df['Category'] = df['Category'].str.strip()  # Clean up whitespace if needed
    df['Message'] = df['Message'].str.strip()
    
    # Load the top words to use in the vectorizer
    top_words = load_top_words()
    
    # Vectorize the text data using only the top words
    vectorizer = CountVectorizer(vocabulary=top_words)
    X = vectorizer.fit_transform(df['Message'])
    y = df['Category']  # Labels: ham or spam
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f'Model Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(cm)
    
    # Save the model and vectorizer for later use
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    return model, vectorizer

def detect_email_type():
    """Detect if the email is spam or ham using the trained model."""
    message = body_text.get("1.0", tk.END).strip()
    
    if not message:
        messagebox.showwarning("Input Error", "Please provide the email content.")
        return
    
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]

    if prediction == "spam":
        messagebox.showerror("Result", "🚨 This Email is SPAM! 🚨")
    else:
        messagebox.showinfo("Result", "✅ This Email is HAM (Safe to Go)! ✅")
    
    clear_fields()

def clear_fields():
    """Clear input fields."""
    body_text.delete("1.0", tk.END)

if __name__ == '__main__':
    print("Training spam detection model with top words...")
    train_spam_model('Training_Dataset.csv')  # Replace with your dataset filename

    # Initialize main window
    root = tk.Tk()
    root.title("Email Spam Detector")
    root.geometry("600x400")
    root.config(bg="#1f1f2e")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#1f1f2e", foreground="#f8f8f8", font=("Helvetica", 12, "bold"))
    style.configure("TButton", font=("Helvetica", 12, "bold"), background="#ff6600", foreground="white")
    style.configure("TText", font=("Helvetica", 12))

    title_label = ttk.Label(root, text="Enter Email Content", font=("Helvetica", 18, "bold"))
    title_label.pack(pady=10)

    body_label = ttk.Label(root, text="Message:")
    body_label.pack(anchor="w", padx=20)
    body_text = tk.Text(root, wrap="word", height=10, width=50, font=("Helvetica", 12))
    body_text.pack(pady=5, padx=20)

    submit_button = ttk.Button(root, text="Submit", command=detect_email_type)
    submit_button.pack(pady=20)

    root.mainloop()