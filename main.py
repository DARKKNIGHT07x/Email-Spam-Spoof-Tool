import os
import pandas as pd
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib  # For saving and loading the model
import tkinter as tk
from tkinter import ttk, messagebox

# Define the scope for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Authenticate the user with Gmail API and return the credentials."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_body(payload):
    """Extract the body of the email, handling both plain text and HTML."""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
            elif part['mimeType'] == 'text/html':
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    elif 'body' in payload and 'data' in payload['body']:
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    return "No Body"

def fetch_emails_with_body(service, max_results=100):
    """Fetch emails and extract the subject, sender, date, and body."""
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])

    email_data = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_data['payload']['headers']
        
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "No Sender")
        date = next((h['value'] for h in headers if h['name'] == 'Date'), "No Date")
        
        body = get_body(msg_data['payload'])

        email_data.append({'Date': date, 'Sender': sender, 'Subject': subject, 'Body': body})
    return email_data

def save_to_csv(email_data, filename='Training_Dataset.csv'):
    """Save emails to a CSV file."""
    df = pd.DataFrame(email_data)
    df.to_csv(filename, index=False)
    print(f"Emails saved to {filename}")

def train_spam_model(email_data):
    """Train a model to detect spam emails."""
    df = pd.DataFrame(email_data)
    df['label'] = df['Subject'].apply(lambda x: 'spam' if 'win' in x.lower() or 'free' in x.lower() or 'congratulations' in x.lower() else 'ham')
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Body'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f'Model Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(cm)
    
    # Save the model and vectorizer for later use
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    return model, vectorizer

def detect_email_type():
    """Detect if the email is spam or ham using the trained model."""
    date = date_entry.get()
    sender = sender_entry.get()
    subject = subject_entry.get()
    body = body_text.get("1.0", tk.END).strip()
    
    email_data = f"{subject} {body}"
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    X = vectorizer.transform([email_data])
    prediction = model.predict(X)[0]

    if prediction == "spam":
        messagebox.showerror("Result", "🚨 This Email is SPAM! 🚨")
    else:
        messagebox.showinfo("Result", "✅ This Email is HAM (Safe to Go)! ✅")
    
    clear_fields()

def clear_fields():
    """Clear input fields."""
    date_entry.delete(0, tk.END)
    sender_entry.delete(0, tk.END)
    subject_entry.delete(0, tk.END)
    body_text.delete("1.0", tk.END)

def main():
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    print("Fetching emails...")
    emails = fetch_emails_with_body(service, max_results=50)
    save_to_csv(emails)
    print("Training spam detection model...")
    train_spam_model(emails)

if __name__ == '__main__':
    main()

    # Initialize main window
    root = tk.Tk()
    root.title("Email Spam Detector")
    root.geometry("600x500")
    root.config(bg="#1f1f2e")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#1f1f2e", foreground="#f8f8f8", font=("Helvetica", 12, "bold"))
    style.configure("TButton", font=("Helvetica", 12, "bold"), background="#ff6600", foreground="white")
    style.configure("TEntry", font=("Helvetica", 12))
    style.configure("TText", font=("Helvetica", 12))

    title_label = ttk.Label(root, text="Enter Email Details", font=("Helvetica", 18, "bold"))
    title_label.pack(pady=10)

    date_label = ttk.Label(root, text="Date:")
    date_label.pack(anchor="w", padx=20)
    date_entry = ttk.Entry(root, width=50)
    date_entry.pack(pady=5, padx=20)

    sender_label = ttk.Label(root, text="Sender:")
    sender_label.pack(anchor="w", padx=20)
    sender_entry = ttk.Entry(root, width=50)
    sender_entry.pack(pady=5, padx=20)

    subject_label = ttk.Label(root, text="Subject:")
    subject_label.pack(anchor="w", padx=20)
    subject_entry = ttk.Entry(root, width=50)
    subject_entry.pack(pady=5, padx=20)

    body_label = ttk.Label(root, text="Body:")
    body_label.pack(anchor="w", padx=20)
    body_text = tk.Text(root, wrap="word", height=10, width=50, font=("Helvetica", 12))
    body_text.pack(pady=5, padx=20)

    submit_button = ttk.Button(root, text="Submit", command=detect_email_type)
    submit_button.pack(pady=20)

    root.mainloop()