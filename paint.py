from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image, ImageDraw, ImageTk
from skimage.io import imread
from skimage.morphology import convex_hull_image
import numpy as np
import torch
import torchvision as tv


class Paint(object):

    DEFAULT_PEN_SIZE = 2
    DEFAULT_COLOR = 'white'
    BACKGROUND_COLOR = 'black'
    RESOLUTION = 600
    IMAGE = Image.new('L', (RESOLUTION, RESOLUTION), BACKGROUND_COLOR)
    BACKGROUND_IMAGE = Image.new('RGB', (RESOLUTION, RESOLUTION), BACKGROUND_COLOR)
    CHANGED = False
    TEMP = 0

    MODEL = torch.nn.Sequential(                #  64
    torch.nn.Conv2d(1, 8, 5),                   #  60 = 64 - (5 - 1)
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),                      #  30 = 60 / 2

    torch.nn.Conv2d(8, 16, 7),                  #  24 = 30 - (7 - 1)
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),                      #  12 = 24 / 2

    torch.nn.Flatten(),
    torch.nn.Linear(16 * 12 * 12, 72),
    torch.nn.ReLU(),
    torch.nn.Linear(72, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 4)
    )

    MODEL.load_state_dict(torch.load('./model/model.pth', map_location=torch.device('cpu')))

    def __init__(self):
        self.root = Tk()
        self.root.configure(bg='#3E4149')
        self.root.title("Shape Recognition (Piotr Wrzosek)")

        self.pen_button = Button(self.root, text='save', command=self.save, highlightbackground='#3E4149', bg='#8c8c8c', fg='#ececec')
        self.pen_button.grid(row=0, column=0)

        self.color_button = Button(self.root, text='color', command=self.choose_color, highlightbackground='#3E4149', bg='#8c8c8c', fg='#ececec')
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='clear', command=self.use_eraser, highlightbackground='#3E4149', bg='#8c8c8c', fg='#ececec')
        self.eraser_button.grid(row=0, column=1)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL, bg='#3E4149', fg='#ececec')
        self.choose_size_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg=self.BACKGROUND_COLOR, width=self.RESOLUTION, height=self.RESOLUTION)
        self.c.grid(row=1, columnspan=4)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.after_id = None
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        # self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def cnn(self):
        class_names = ['circle', 'triangle', 'square', 'triangle_flip']
        DATA = torch.empty(1, 1, 64, 64)
        DATA[0] = tv.transforms.functional.to_tensor(self.IMAGE)
        ACTIVATION = self.MODEL(DATA)
        VALUE = torch.argmax(ACTIVATION, 1)
        return class_names[VALUE[0]]


    def analyze(self):
        ### prepare self.IMAGE
        self.IMAGE.save('./data/image.png', 'png')
        img = imread('./data/image.png')
        threshold = 0.5
        img[img <= threshold] = 0
        img[img > threshold] = 1
        chull = convex_hull_image(img)
        position = Image.fromarray((chull*255).astype(np.uint8)).getbbox()
        cropped = Image.fromarray((img*255).astype(np.uint8), mode = 'L').crop(position)
        cropped = cropped.resize((64, 64))
        cropped.save('./data/image_crop.png', 'png')
        self.IMAGE = Image.open('./data/image_crop.png')

        ### use neural network to get shape_type
        shape_type = self.cnn()
        return shape_type, position

    def add_circle(self, position):
        draw = ImageDraw.Draw(self.BACKGROUND_IMAGE)
        draw.ellipse(position, outline = self.color, width = self.line_width)

    def add_square(self, position):
        draw = ImageDraw.Draw(self.BACKGROUND_IMAGE, "RGB")
        draw.rectangle(position, outline = self.color, width = self.line_width)

    def add_triangle(self, position):
        draw = ImageDraw.Draw(self.BACKGROUND_IMAGE)
        ax = position[0]
        ay = position[3]
        bx = position[2]
        by = position[3]
        cx = position[0] + 0.5 * (position[2] - position[0])
        cy = position[1]
        points = (
            (ax, ay),
            (bx, by),
            (cx, cy),
            (ax, ay),
            (bx, by)
        )
        draw.line(points, fill = self.color, width = self.line_width, joint = 'curve')

    def add_triangle_flip(self, position):
        draw = ImageDraw.Draw(self.BACKGROUND_IMAGE)
        ax = position[0]
        ay = position[1]
        bx = position[2]
        by = position[1]
        cx = position[0] + 0.5 * (position[2] - position[0])
        cy = position[3]
        points = (
            (ax, ay),
            (bx, by),
            (cx, cy),
            (ax, ay),
            (bx, by)
        )
        draw.line(points, fill = self.color, width = self.line_width, joint = 'curve')

    def add_shape(self, shape_type, position):
        if shape_type == 'circle':
            self.add_circle(position)
        elif shape_type == 'square':
            self.add_square(position)
        elif shape_type == 'triangle':
            self.add_triangle(position)
        elif shape_type == 'triangle_flip':
            self.add_triangle_flip(position)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def capture_image(self):
        if self.CHANGED:
            shape_type, position = self.analyze()
            self.add_shape(shape_type, position)
            self.TEMP = ImageTk.PhotoImage(self.BACKGROUND_IMAGE)
            self.c.create_image(0, 0, anchor = "nw", image = self.TEMP)
            self.IMAGE = Image.new('L', (self.RESOLUTION, self.RESOLUTION), self.BACKGROUND_COLOR)
            self.CHANGED = False

    def auto_capture(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(670, self.capture_image)

    def save(self):
        self.BACKGROUND_IMAGE.save("image.png", "png")

    def use_eraser(self):
        color = self.color
        self.c.delete('all')
        self.color = color
        self.IMAGE = Image.new('L', (self.RESOLUTION, self.RESOLUTION), self.BACKGROUND_COLOR)
        self.BACKGROUND_IMAGE = Image.new('RGB', (self.RESOLUTION, self.RESOLUTION), self.BACKGROUND_COLOR)
        self.CHANGED = False

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def reset(self, event):
        self.old_x, self.old_y = None, None    

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = self.BACKGROUND_COLOR if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            draw = ImageDraw.Draw(self.IMAGE)
            draw.line(((self.old_x, self.old_y), (event.x,event.y)), 'white', width=self.DEFAULT_PEN_SIZE)
            self.CHANGED = True
        self.old_x = event.x
        self.old_y = event.y
        self.auto_capture()


if __name__ == '__main__':
    Paint()