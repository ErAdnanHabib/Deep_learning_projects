import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('age_gen_detection.keras')

def detect(file_path):
    image = Image.open(file_path)
    image = image.resize((48, 48))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    age = int(np.round(pred[1][0]))
    sex = int(np.round(pred[0][0]))
    sex_f = ['Male', 'Female']
    label1.config(text=f"Age: {age}")
    label2.config(text=f"Gender: {sex_f[sex]}")
    return age, sex_f[sex]

def show_detect_button(file_path):
    detect_button = tk.Button(top, text="Detect image", command=lambda: detect(file_path), padx=10, pady=5)
    detect_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        label2.configure(text='')
        show_detect_button(file_path)
    except FileNotFoundError:
        print("No file selected")

top = tk.Tk()
top.geometry('800x600')
top.title('AGE AND GENDER DETECTION USING DEEP LEARNING')
top.configure(background='#CDCDCD')

heading = tk.Label(top, text="AGE AND GENDER DETECTION", pady=20, font=('arial', 20, "bold"))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

sign_image = tk.Label(top)
sign_image.pack(side='bottom', expand=True)

label1 = tk.Label(top, background='#CDCDCD', font=('arial', 15, "bold"))
label1.pack(side='bottom', expand=True)

label2 = tk.Label(top, background='#CDCDCD', font=('arial', 15, "bold"))
label2.pack(side='bottom', expand=True)

upload_button = tk.Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 10, "bold"))
upload_button.pack(side='bottom', pady=50)

top.mainloop()