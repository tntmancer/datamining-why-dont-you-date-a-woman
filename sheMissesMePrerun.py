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
        
        # Precompute facial descriptors for all images
        print("Loading facial descriptors...")
        self.face_descriptors = self.load_or_compute_descriptors()
        print(f"Loaded descriptors for {len(self.face_descriptors)} images")
        
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
        # Update weights for similar images
        self.update_weights(self.current_image_path, "like")
        self.next_image()
    
    def pass_face(self):
        print(f"Passed: {self.current_image_path}")
        print(f"Total Passed Images: {len(self.disliked_images) + 1}")
        print(f"Total Evaluated Images: {len(self.liked_images) + len(self.disliked_images) + 1}")
        self.disliked_images.append(self.current_image_path)
        # Update weights for similar images  
        self.update_weights(self.current_image_path, "pass")
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

    def update_weights(self, image_path, action, k=5):
        """ Update the weights of similar images based on user feedback. """
        similar_images = self.get_similar_images(image_path, k)
        
        # Create a new heap with updated weights
        new_heap = []
        for weight, index, path in self.image_heap:
            if path in similar_images:
                # Lower weight = higher priority
                if action == "like":
                    new_weight = weight - 1  
                # Higher weight = lower priority
                elif action == "pass":
                    new_weight = weight + 1  
                else:
                    new_weight = weight
                heapq.heappush(new_heap, (new_weight, index, path))
            else:
                heapq.heappush(new_heap, (weight, index, path))
        
        self.image_heap = new_heap

    # This saves us from recalculating this at runtime each time
    def load_or_compute_descriptors(self):
        """ Load descriptors from cache or compute them if cache doesn't exist or is outdated. """
        cache_file = "face_descriptors_cache.pkl"
        
        # Check if cache exists and is newer than the faces directory
        if os.path.exists(cache_file):
            try:
                cache_time = os.path.getmtime(cache_file)
                faces_time = os.path.getmtime("faces")
                
                if cache_time > faces_time:
                    print("Loading descriptors from cache...")
                    with open(cache_file, 'rb') as f:
                        cached_descriptors = pickle.load(f)
                    
                    # Verify all current images are in cache
                    if all(img_path in cached_descriptors for img_path in self.image_paths):
                        return {path: cached_descriptors[path] for path in self.image_paths}
                    else:
                        print("Cache missing some images, recomputing...")
                else:
                    print("Cache is outdated, recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}, recomputing...")
        
        # Compute descriptors and save to cache
        print("Computing facial descriptors...")
        descriptors = self.precompute_descriptors()
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(descriptors, f)
            print(f"Saved descriptors to {cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
        return descriptors

    def precompute_descriptors(self):
        """ Precompute facial descriptors for all images to speed up similarity calculations. """
        descriptors = {}
        total_images = len(self.image_paths)
        
        for i, img_path in enumerate(self.image_paths):
            # print(f"Processing {i+1}/{total_images}: {img_path}")
            try:
                img = Image.open(img_path)
                img_desc = self.face_recognizer.recognize_faces(np.array(img))
                # Store first face descriptor
                if img_desc:
                    descriptors[img_path] = img_desc[0]  
                else:
                    print(f"  No face found in {img_path}")
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
        
        return descriptors

    def get_similar_images(self, image_path, k):
        """ Get k most similar images based on face descriptors. """
        # Get the reference descriptor from precomputed cache
        if image_path not in self.face_descriptors:
            return []
        
        ref_descriptor = self.face_descriptors[image_path]
        
        # Calculate similarity with all other images using precomputed descriptors
        similarities = []
        for img_path, descriptor in self.face_descriptors.items():
            if img_path != image_path:  # Don't compare with itself
                similarity = self.face_recognizer.face_similarity(ref_descriptor, descriptor)
                similarities.append((similarity, img_path))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0])
        return [img[1] for img in similarities[:k]]

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