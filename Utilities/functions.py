import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def select_file():
    print("button_2 clicked")
    file_path = filedialog.askopenfilename()
    if file_path:
        messagebox.showinfo("Video Loaded", "Video successfully loaded for analysis.")
    return file_path

def start_prediction(path):  
    print("button_1 clicked")
    with open(path) as file:
        exec(file.read())  