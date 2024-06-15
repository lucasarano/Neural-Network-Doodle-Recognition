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
        self.btn_predict = tk.Button(root, text='Predict', command=self.predict)
        self.btn_predict.pack()
        self.label = tk.Label(root, text='Draw a digit and click Predict')
        self.label.pack()

    def paint(self, event):
        # Increase the size of the oval and the line width
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')


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
        plt.imshow(self.image, cmap='gray')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
