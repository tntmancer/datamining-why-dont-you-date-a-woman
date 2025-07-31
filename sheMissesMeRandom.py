import tkinter as tk
from PIL import Image, ImageTk
import os
import heapq
from faceRecognizer import FaceRecognizer
import random
import numpy as np
import pickle

class FaceRatingApp:
    def __init__(self, image_paths):
        # models taken from dlib examples
        self.face_recognizer = FaceRecognizer('models/encoderModel.dat', 'models/recognitionModel.dat')
        self.root = tk.Tk()
        self.root.title("She Misses Me")
        self.image_paths = image_paths.copy()
        random.shuffle(self.image_paths)
        self.image_heap = []
        self.liked_images = []
        self.disliked_images = []
        self.create_image_heap()
        self.current_image_path = None
        
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.like_btn = tk.Button(button_frame, text="Like ❤️", 
             command=self.like_face, bg="green",
             width=10, height=2, font=('Arial', 12, 'bold'), 
             justify=tk.CENTER)
        self.like_btn.pack(side=tk.LEFT, padx=10)
        
        self.pass_btn = tk.Button(button_frame, text="Pass ❌", 
                                 command=self.pass_face, bg="red",
                                 width=10, height=2, font=('Arial', 12, 'bold'),
                                 justify=tk.CENTER)
        self.pass_btn.pack(side=tk.LEFT, padx=10)
        
        self.next_image()
    
    def show_current_image(self):
        if self.current_image_path:
            # Load and resize image
            img = Image.open(self.current_image_path)
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
    
    def like_face(self):
        print(f"Liked: {self.current_image_path}")
        print(f"Total Liked Images: {len(self.liked_images) + 1}")
        print(f"Total Evaluated Images: {len(self.liked_images) + len(self.disliked_images) + 1}")
        self.liked_images.append(self.current_image_path)
        self.next_image()
    
    def pass_face(self):
        print(f"Passed: {self.current_image_path}")
        print(f"Total Passed Images: {len(self.disliked_images) + 1}")
        print(f"Total Evaluated Images: {len(self.liked_images) + len(self.disliked_images) + 1}")
        self.disliked_images.append(self.current_image_path)
        self.next_image()

    def next_image(self):
        """ Move to the next image in the heap. """
        if self.image_heap:
            # Get the next image with the highest priority (lowest weight)
            next_image = heapq.heappop(self.image_heap)
            # Get path from 3-tuple (weight, index, path)
            self.current_image_path = next_image[2]  
            self.show_current_image()
        else:
            self.root.quit()

    def create_image_heap(self):
        """ Initialize the heap with images and their weights. """
        # enumerate is done here because otherwise ties are broken alphabetically, want random order
        for i, image_path in enumerate(self.image_paths):
            heapq.heappush(self.image_heap, (0, i, image_path))

    def run(self):
        self.root.mainloop()

image_paths = ["faces/" + f for f in os.listdir("faces") if f.endswith((".jpg", ".png"))]

app = FaceRatingApp(image_paths)
app.run()