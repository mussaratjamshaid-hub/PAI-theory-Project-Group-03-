import tkinter as tk
import pickle

vectorizer = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model_lr.pkl","rb"))

def check_spam():
    text = entry.get("1.0", tk.END)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]

    if result == 1:
        label_result.config(text="Spam Email ❌")
    else:
        label_result.config(text="Not Spam ✅")

root = tk.Tk()
root.title("Spam Classifier")

entry = tk.Text(root, height=10, width=50)
entry.pack()

btn = tk.Button(root, text="Check Email", command=check_spam)
btn.pack()

label_result = tk.Label(root, text="", font=("Arial", 14))
label_result.pack()

root.mainloop()
