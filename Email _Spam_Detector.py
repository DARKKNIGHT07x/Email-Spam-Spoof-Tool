import tkinter as tk
from tkinter import ttk, messagebox
import joblib  # Assuming you'll load a trained ML model

# Load your trained ML model (replace 'spam_model.pkl' with your actual model file)
# model = joblib.load('spam_model.pkl')  # Uncomment and use your model file path

# Function to detect if the email is spam or ham using your ML model
def detect_email_type():
    # Get input data
    date = date_entry.get()
    sender = sender_entry.get()
    subject = subject_entry.get()
    body = body_text.get("1.0", tk.END).strip()

    # Preprocess input (this part should match your ML model's input requirements)
    email_data = f"{subject} {body}"  # Combine subject and body as a single input
    # prediction = model.predict([email_data])[0]  # Use your ML model to predict

    # Mock Prediction (Replace this with actual prediction logic)
    # Simulate spam detection (for demo purposes)
    if "offer" in email_data.lower() or "win" in email_data.lower():
        prediction = "Spam"
    else:
        prediction = "Ham"

    # Show result
    if prediction == "Spam":
        messagebox.showerror("Result", "🚨 This Email is SPAM! 🚨")
    else:
        messagebox.showinfo("Result", "✅ This Email is HAM (Safe to Go)! ✅")
    
    # Clear fields after submission
    clear_fields()

# Function to clear input fields
def clear_fields():
    date_entry.delete(0, tk.END)
    sender_entry.delete(0, tk.END)
    subject_entry.delete(0, tk.END)
    body_text.delete("1.0", tk.END)

# Initialize main window
root = tk.Tk()
root.title("Email Spam Detector")
root.geometry("600x500")
root.config(bg="#1f1f2e")

# Styling
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#1f1f2e", foreground="#f8f8f8", font=("Helvetica", 12, "bold"))
style.configure("TButton", font=("Helvetica", 12, "bold"), background="#ff6600", foreground="white")
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TText", font=("Helvetica", 12))

# Title
title_label = ttk.Label(root, text="Enter Email Details", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

# Date Input
date_label = ttk.Label(root, text="Date:")
date_label.pack(anchor="w", padx=20)
date_entry = ttk.Entry(root, width=50)
date_entry.pack(pady=5, padx=20)

# Sender Input
sender_label = ttk.Label(root, text="Sender:")
sender_label.pack(anchor="w", padx=20)
sender_entry = ttk.Entry(root, width=50)
sender_entry.pack(pady=5, padx=20)

# Subject Input
subject_label = ttk.Label(root, text="Subject:")
subject_label.pack(anchor="w", padx=20)
subject_entry = ttk.Entry(root, width=50)
subject_entry.pack(pady=5, padx=20)

# Body Input
body_label = ttk.Label(root, text="Body:")
body_label.pack(anchor="w", padx=20)
body_text = tk.Text(root, wrap="word", height=10, width=50, font=("Helvetica", 12))
body_text.pack(pady=5, padx=20)

# Submit Button
submit_button = ttk.Button(root, text="Submit", command=detect_email_type)
submit_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
