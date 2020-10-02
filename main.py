import cv2
import tkinter as tk
import os
from PIL import Image, ImageTk
import numpy as np

from model import Model


def draw(event):
    thickness = 6
    x_motion, y_motion = event.x, event.y
    canvas.create_oval(x_motion-thickness, y_motion-thickness, x_motion+thickness, y_motion+thickness, fill='black', outline='black')


# TODO: zmień widok przyczisku aktywnego/nieaktywnego
def pencil_butt():
    pencil_button.config(bg='gray', bd=1)
    root.unbind_class('Canvas', '<B1-Motion>')
    root.unbind_class('Canvas', '<Button-1>')
    root.bind_class('Canvas', '<B1-Motion>', draw)
    root.bind_class('Canvas', '<Button-1>', draw)


# TODO: zmień widok przyczisku aktywnego/nieaktywnego
def erase(event):
    thickness = 15
    x_motion, y_motion = event.x, event.y
    canvas.create_oval(x_motion-thickness, y_motion-thickness, x_motion+thickness, y_motion+thickness, fill='white', outline='white')


def eraser_butt():
    eraser_button.config(bg='gray', bd=1)
    root.unbind_class('Canvas', '<B1-Motion>')
    root.unbind_class('Canvas', '<Button-1>')
    root.bind_class('Canvas', '<B1-Motion>', erase)
    root.bind_class('Canvas', '<Button-1>', erase)


def clear_canvas():
    canvas.delete("all")


def canvas_predict():
    canvas.postscript(file="temp/canvas_obj.eps")
    img = Image.open("temp/canvas_obj.eps")
    img.save("temp/canvas.jpeg", "JPEG")
    prediction = model.make_predict()
    if np.max(prediction) > 0.7:
        predict_result.config(text='wygląda jak {} ({}%)'.format(np.argmax(prediction), round(np.max(prediction)*100, 2)))
    else:
        predict_result.config(text='spróbuj jeszcze raz')
    os.remove("temp/canvas_obj.eps")
    os.remove("temp/canvas.jpeg")


if __name__ == "__main__":

    model = Model(model_path='./models/handwritten_digits.model',
                img_path='./temp/canvas.jpeg')

    root = tk.Tk()
    root.title('SmartCanvas - Digit Edition')
    root.resizable(False, False)


    eraser_img = tk.PhotoImage(file='./icons/eraser.png')
    pencil_img = tk.PhotoImage(file='./icons/pencil.png')


    frame = tk.Frame(root, height=475, width=426, bg='lightgray')
    frame.pack()

    clear_button = tk.Button(frame, command=clear_canvas, text='clear', bg='#B9B9B9', activebackground='red', cursor='exchange')
    clear_button.place(relx=0.03, rely=0.05)

    predict_button = tk.Button(frame, command=canvas_predict, text='predict', bg='#B9B9B9', activebackground='lime')
    predict_button.place(relx=0.18, rely=0.05)

    eraser_button = tk.Button(frame, command=eraser_butt, image=eraser_img, bg='#B9B9B9', bd=2)
    eraser_button.place(relx=0.38, rely=0.05)

    pencil_button = tk.Button(frame, command=pencil_butt, image=pencil_img, bg='#B9B9B9', bd=2)
    pencil_button.place(relx=0.47, rely=0.05)

    predict_result = tk.Label(frame, text='prediction', bg='#F8D861', width=24)
    predict_result.place(relx=0.568, rely=0.06)

    canvas = tk.Canvas(frame, height=400, width=400, bg='white', cursor='tcross')
    canvas.place(relx=0.03, rely=0.13)


    root.mainloop()

