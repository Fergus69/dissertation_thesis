import os
import shutil

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import transforms
import timm 
from pathlib import Path
from tkinter import Button, Canvas, PhotoImage, Text, Tk, filedialog, messagebox
import tkinter as tk


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path('./assets/frame0')

device_pt = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_timm_model(checkpoint_path):
    model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=False, num_classes=2)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device_pt, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device_pt)
        model.eval() 
        print(f"Model timm încărcat cu succes pe {device_pt}")
        return model
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        return None

model_pt = load_timm_model("./best_model.pth")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def select_file():
    print("button_2 clicked")
    suffix=None
    file_path = filedialog.askopenfilename()
    if file_path:
        suffix=Path(file_path).suffix
        file_extension = Path(file_path).suffix
        if file_extension == '.jpg':
            messagebox.showinfo("Photo Loaded", "Photo successfully loaded for analysis.")
            return file_path, suffix
        else:
             messagebox.showinfo("Wrong type", "This application only supports jpg files.")    
    return None
    
def count_items_in_folder(folder_path):
    """ Counts the number of items in the given folder. """
    try:
        items = os.listdir(folder_path)
        return len(items)
    except FileNotFoundError:
        print("The folder does not exist.")
        return 0
    except PermissionError:
        print("Permission denied for accessing the folder.")
        return 0


def resize_images_in_same_folder(folder_path, size=(256, 256)):
    """
    Resize all images in the specified folder to the given size and save them back to the same folder.

    Parameters:
    - folder_path: Path to the folder containing the images to resize.
    - size: A tuple specifying the new size as (width, height).
    """
    folder_path = Path(folder_path)

    for file in folder_path.iterdir():
        if file.is_file() and file.suffix in ['.jpg', '.jpeg', '.png']:
            try:
                img = Image.open(file)
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(file)
                print(f"Resized and saved {file.name}")
            except Exception as e:
                print(f"Failed to resize {file.name}. Reason: {e}")


def delete_image():
    canvas.delete(image_2) 



def create_image(folder_path):
    folder_path = Path(folder_path)
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            path = str(file)
            try:
                img_pil = Image.open(path)
                
                poza = ImageTk.PhotoImage(img_pil)
                
                global image_2 
                image_2 = canvas.create_image(
                    581.0,
                    377.0,
                    image=poza
                )
                
                canvas.image = poza 
                
                print(f"Displayed {file.name} on canvas.")
                break 
            except Exception as e:
                print(f"Failed to display {file.name}. Reason: {e}")
                continue

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

picture_folder = './build/assets/picture'
predict_folder = './build/assets/predict/predict'

def button_2_func():
    os.makedirs(picture_folder, exist_ok=True)
    os.makedirs(predict_folder, exist_ok=True)
    
    clear_directory(picture_folder)
    clear_directory(predict_folder)
    
    result = select_file()
    if result:
        fisier, suffix = result 
        filename = os.path.basename(fisier) 
        shutil.copy(fisier, os.path.join(picture_folder, filename))
        shutil.copy(fisier, os.path.join(predict_folder, filename))
        resize_images_in_same_folder(picture_folder, (543, 635))
        delete_image()
        create_image(picture_folder)
        textbox.configure(state='normal')
        textbox.delete('1.0', tk.END)
        textbox.configure(state='disabled')



def alg():
    if model_pt is None:
        messagebox.showerror("Eroare", "Modelul nu este încărcat!")
        return

    crop_size = 224 
    
    tta_preprocess = transforms.Compose([
        transforms.FiveCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                transforms.ToTensor()(crop)
            ) for crop in crops
        ]))
    ])

    img_path = None
    for file in Path(predict_folder).iterdir():
        if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img_path = str(file)
            break
    
    if not img_path:
        messagebox.showwarning("Atenție", "Încarcă o imagine mai întâi.")
        return

    try:
        img = Image.open(img_path).convert('RGB')
        
        if img.width < crop_size or img.height < crop_size:
            messagebox.showerror("Eroare", f"Imaginea este prea mică pentru analiza TTA ({img.width}x{img.height}).")
            return

        input_batch = tta_preprocess(img).to(device_pt) 

        with torch.no_grad():
            outputs = model_pt(input_batch) 
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            prob_ai = probs.mean().item()

        if prob_ai > 0.5:
            text = f"AI (TTA: {prob_ai * 100:.1f}%)"
        else:
            text = f"REAL (TTA: {(1 - prob_ai) * 100:.1f}%)"

        textbox.configure(state='normal')
        textbox.delete('1.0', tk.END)
        textbox.insert(tk.END, text)
        textbox.configure(state='disabled')

    except Exception as e:
        messagebox.showerror("Eroare", f"Analiza a eșuat: {e}")

window = Tk()

window.geometry("1161x946")
window.configure(bg = "#FFFFFF")


image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 946,
    width = 1161,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_1 = canvas.create_image(
    581.0,
    473.0,
    image=image_image_1,
    anchor='center'
)



button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda :alg(),  
    relief="flat"
)
button_1.place(
    x=309.0,
    y=826.0,
    width=543.0,
    height=70.0
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    581.0,
    377.0,
    image=image_image_2
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: button_2_func(),
    relief="flat"
)
button_2.place(
    x=512.0,
    y=711.0,
    width=139.0,
    height=30.0
)


textbox = Text(
    window,
    height=1,
    width=30,
    bg="#FFFFFF",
    fg="black",
    borderwidth=1,
    relief="solid"
)
textbox.place(x=460, y=750)  
textbox.configure(state='disabled') 


window.resizable(False, False)
window.mainloop()



