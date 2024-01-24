import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import threading
import time
import torch
import torchvision.transforms as transforms 
import AdaIN_net as net
import os

print ("Initializing...")

image_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load('models/encoder.pth', map_location='cpu'))
decoder = net.encoder_decoder.decoder
decoder.load_state_dict(torch.load('models/decoder.pth', map_location='cpu'))
model = net.AdaIN_net(encoder, decoder)

model.to(device=device)
model.eval()

style_stary_night = Image.open('images/style/starynight.jpg', mode='r').convert('RGB')
style_Andy_Warhol_97 = Image.open('images/style/Andy_Warhol_97.jpg', mode='r').convert('RGB')
style_brushstrokes = Image.open('images/style/brushstrokes.jpg', mode='r').convert('RGB')
style_chagall_marc_1 = Image.open('images/style/chagall_marc_1.jpg', mode='r').convert('RGB')
style_the_persistence_of_memory_1931 = Image.open('images/style/the-persistence-of-memory-1931.jpg', mode='r').convert('RGB')

style_stary_night = transforms.Resize(size=image_size)(style_stary_night)
style_Andy_Warhol_97 = transforms.Resize(size=image_size)(style_Andy_Warhol_97)
style_brushstrokes = transforms.Resize(size=image_size)(style_brushstrokes)
style_chagall_marc_1 = transforms.Resize(size=image_size)(style_chagall_marc_1)
style_the_persistence_of_memory_1931 = transforms.Resize(size=image_size)(style_the_persistence_of_memory_1931)

style_stary_night_tensor = transforms.ToTensor()(style_stary_night).unsqueeze(0)
style_Andy_Warhol_97_tensor = transforms.ToTensor()(style_Andy_Warhol_97).unsqueeze(0)
style_brushstrokes_tensor = transforms.ToTensor()(style_brushstrokes).unsqueeze(0)
style_chagall_marc_1_tensor = transforms.ToTensor()(style_chagall_marc_1).unsqueeze(0)
style_the_persistence_of_memory_1931_tensor = transforms.ToTensor()(style_the_persistence_of_memory_1931).unsqueeze(0)

styles = [style_stary_night_tensor, style_Andy_Warhol_97_tensor, style_brushstrokes_tensor, style_chagall_marc_1_tensor, style_the_persistence_of_memory_1931_tensor]
style_names = ["Starry Night", "Andy Warhol", "Brushstrokes", "Chagall Marc", "Persistence of Memory"]
alphas = [0.1, 0.5, 0.9]

print('GUI Initializing...')

class Application:
    def __init__(self, window, window_title, video_source=0, initial_image_size=(150, 150)):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.initial_image_size = initial_image_size
        self.vid = cv2.VideoCapture(video_source)
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        self.delay = 15
        self.update()
        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            file_path = "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"
            cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = transforms.Resize(size=image_size)(frame)
            frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)

            new_window = tk.Toplevel(self.window)
            new_window.title("Stylized Frames")
            self.stylized_images = []
            self.labels = []

            for i, style_tensor in enumerate(styles):
                for j, alpha in enumerate(alphas):
                    with torch.no_grad():
                        stylized_frame = model(frame_tensor.to(device), style_tensor.to(device), alpha)
                    stylized_frame = stylized_frame.squeeze(0).cpu().detach()
                    original_stylized_frame = transforms.ToPILImage()(stylized_frame)
                    resized_stylized_frame = original_stylized_frame.resize(self.initial_image_size, Image.ANTIALIAS)
                    stylized_photo = ImageTk.PhotoImage(image=resized_stylized_frame)
                    label = tk.Label(new_window, image=stylized_photo)
                    label.image = stylized_photo
                    label.grid(row=i, column=j)
                    tk.Label(new_window, text=f"{style_names[i]} Alpha {alpha}").grid(row=i, column=j, sticky="S")
                    self.stylized_images.append((i, j, original_stylized_frame))
                    self.labels.append(label)
            os.remove(file_path)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

print ('Done!')


app = Application(tk.Tk(), "Tkinter and OpenCV", initial_image_size=(300, 200))
