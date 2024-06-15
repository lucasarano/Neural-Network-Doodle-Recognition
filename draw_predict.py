import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Check for available device: CUDA, MPS, or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load the trained model
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.eval()

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.btn_predict = tk.Button(root, text='Predict', command=self.predict)
        self.btn_predict.pack(side=tk.LEFT)
        self.btn_erase = tk.Button(root, text='Erase', command=self.erase)
        self.btn_erase.pack(side=tk.LEFT)
        self.label = tk.Label(root, text='Draw a digit and click Predict')
        self.label.pack()
        self.last_x, self.last_y = None, None

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), fill='black', width=10)
            self.draw.line((self.last_x, self.last_y, event.x, event.y), fill='black', width=10)
        self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def erase(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text='Draw a digit and click Predict')

    def preprocess_image(self):
        image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        image = ImageOps.invert(image)
        image = image.convert('L')
        image = np.array(image).astype(np.float32)
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        image = image / 255.0
        return image.to(device)

    def predict(self):
        img = self.preprocess_image()
        output = model(img)
        prediction = output.argmax(dim=1, keepdim=True).item()
        self.label.config(text=f'Prediction: {prediction}')
        print(f"Prediction: {prediction}")
        # plt.imshow(self.image, cmap='gray')
        # plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
