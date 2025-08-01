# Generate a graph starting from an image
# The image should show the most similar and most dissimilar face to that image
# This should be a binary tree with a varible depth
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from faceRecognizer import FaceRecognizer
import random
import numpy as np
import pickle

class FaceTreeNode:
    def __init__(self, image_path, descriptor):
        self.image_path = image_path
        self.descriptor = descriptor
        self.left = None  # Most similar face
        self.right = None  # Most dissimilar face

class FaceGraph:
    def __init__(self, image_paths, depth=3):
        self.root = None
        self.image_paths = image_paths.copy()  # Keep original intact
        self.available_images = set(image_paths)  # Track available images
        self.depth = depth
        self.face_recognizer = FaceRecognizer('models/encoderModel.dat', 'models/recognitionModel.dat')
        self.face_descriptors = self.load_or_compute_descriptors()
        self.create_graph()
    
    def create_graph(self):
        """ Initialize the graph with a random image as the root and build recursively. """
        if not self.image_paths:
            return None

        # Select a random image as the root
        root_image = random.choice(list(self.available_images))
        root_descriptor = self.face_descriptors.get(root_image)
        self.root = FaceTreeNode(root_image, root_descriptor)
        # Remove the root image from available images
        self.available_images.remove(root_image)

        # Build the graph recursively
        self._build_graph(self.root, depth=self.depth)

    def _build_graph(self, node, depth):
        """ Recursively build the graph to the specified depth. """
        if depth == 0 or len(self.available_images) < 2:
            return

        # Find the most similar and dissimilar images
        most_similar = self.find_most_similar(node.image_path)
        most_dissimilar = self.find_most_dissimilar(node.image_path)

        # Only create nodes if we found valid images
        if most_similar and most_similar in self.available_images:
            node.left = FaceTreeNode(most_similar, self.face_descriptors.get(most_similar))
            self.available_images.remove(most_similar)
            self._build_graph(node.left, depth - 1)
        
        if most_dissimilar and most_dissimilar in self.available_images:
            node.right = FaceTreeNode(most_dissimilar, self.face_descriptors.get(most_dissimilar))
            self.available_images.remove(most_dissimilar)
            self._build_graph(node.right, depth - 1)

    def find_most_similar(self, image_path):
        """ Find the most similar image to the given image path from available images. """
        target_descriptor = self.face_descriptors.get(image_path)
        if target_descriptor is None or not self.available_images:
            return None

        # Calculate distances to all available images
        distances = []
        for img_path in self.available_images:
            if img_path != image_path:
                desc = self.face_descriptors.get(img_path)
                if desc is not None:
                    distance = self.face_recognizer.face_similarity(target_descriptor, desc)
                    distances.append((distance, img_path))

        if not distances:
            return None

        # Get the image with the smallest distance
        most_similar_image = min(distances, key=lambda x: x[0])[1]
        return most_similar_image
    
    def find_most_dissimilar(self, image_path):
        """ Find the most dissimilar image to the given image path from available images. """
        target_descriptor = self.face_descriptors.get(image_path)
        if target_descriptor is None or not self.available_images:
            return None

        # Calculate distances to all available images
        distances = []
        for img_path in self.available_images:
            if img_path != image_path:
                desc = self.face_descriptors.get(img_path)
                if desc is not None:
                    distance = self.face_recognizer.face_similarity(target_descriptor, desc)
                    distances.append((distance, img_path))

        if not distances:
            return None

        # Get the image with the largest distance
        most_dissimilar_image = max(distances, key=lambda x: x[0])[1]
        return most_dissimilar_image
    
    def load_or_compute_descriptors(self):
        """ Load descriptors from cache or compute them if cache doesn't exist. """
        cache_file = "face_descriptors_cache.pkl"
        
        # Check if cache exists
        if os.path.exists(cache_file):
            try:
                print("Loading descriptors from cache...")
                with open(cache_file, 'rb') as f:
                    cached_descriptors = pickle.load(f)
                    
                    # Verify all current images are in cache
                    if all(img_path in cached_descriptors for img_path in self.image_paths):
                        print(f"Successfully loaded {len(cached_descriptors)} descriptors from cache")
                        return {path: cached_descriptors[path] for path in self.image_paths}
                    else:
                        print("Cache missing some images, recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}, recomputing...")
        else:
            print("No cache file found, computing descriptors...")
        
        # Compute descriptors and save to cache
        print("Computing facial descriptors...")
        descriptors = self.precompute_descriptors()
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(descriptors, f)
            print(f"Saved {len(descriptors)} descriptors to {cache_file}")
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
    
    def display_graph(self):
        """ Display the graph starting from the root node. """
        if not self.root:
            print("No graph to display.")
            return
        
        self._display_node(self.root, depth=0)

    def _display_node(self, node, depth):
        """ Recursively display the graph starting from the given node. """
        if node is None:
            return

        # Display the current node
        print("  " * depth + f"Node: {node.image_path}")

        # Recursively display child nodes
        self._display_node(node.left, depth + 1)
        self._display_node(node.right, depth + 1)
        return
    
class FaceGraphDisplay:
    def __init__(self, root_window, face_graph):
        self.root_window = root_window
        self.face_graph = face_graph
        self.canvas = None
        self.setup_display()
    
    def setup_display(self):
        """ Setup the display window and canvas. """
        self.root_window.title("Face Similarity Graph")
        self.root_window.geometry("1200x800")
        
        # Create scrollable canvas
        frame = tk.Frame(self.root_window)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(frame, bg="white")
        scrollbar_v = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        scrollbar_h = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        scrollbar_v.pack(side="right", fill="y")
        scrollbar_h.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Display the graph
        self.display_graph()
    
    def display_graph(self):
        """ Display the graph with images on the canvas. """
        if not self.face_graph.root:
            self.canvas.create_text(400, 300, text="No graph to display", 
                                  font=("Arial", 16), fill="red")
            return
        
        # Calculate positions and draw the tree
        self._draw_tree(self.face_graph.root, 600, 50, 300, 0)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _draw_tree(self, node, x, y, x_offset, depth):
        """ Recursively draw the tree nodes with images. """
        if node is None:
            return
        
        # Load and resize image
        try:
            img = Image.open(node.image_path)
            img = img.resize((100, 100), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Draw image
            self.canvas.create_image(x, y, image=photo, anchor="center")
            
            # Keep reference to prevent garbage collection
            if not hasattr(self, 'images'):
                self.images = []
            self.images.append(photo)
            
            # Draw image border
            self.canvas.create_rectangle(x-50, y-50, x+50, y+50, outline="black", width=2)
            
            # Draw filename below image
            filename = os.path.basename(node.image_path)
            self.canvas.create_text(x, y+65, text=filename, font=("Arial", 8), anchor="center")
            
            # Draw depth indicator
            if depth == 0:
                label = "ROOT"
            else:
                label = f"Level {depth}"
            self.canvas.create_text(x, y-65, text=label, font=("Arial", 10, "bold"), 
                                  fill="blue", anchor="center")
            
        except Exception as e:
            # Placeholder if load fails
            self.canvas.create_rectangle(x-50, y-50, x+50, y+50, fill="lightgray", outline="black")
            self.canvas.create_text(x, y, text="Error", font=("Arial", 10), anchor="center")
        
        # Calculate positions for children
        y_child = y + 150
        x_left = x - x_offset
        x_right = x + x_offset
        new_x_offset = max(x_offset // 2, 100)  # Minimum offset of 100
        
        # Draw connections and child nodes
        if node.left:
            # Draw line to left child (most similar)
            self.canvas.create_line(x, y+50, x_left, y_child-50, fill="green", width=2)
            self.canvas.create_text((x + x_left) // 2, (y+50 + y_child-50) // 2, 
                                  font=("Arial", 8, "bold"), fill="green")
            self._draw_tree(node.left, x_left, y_child, new_x_offset, depth + 1)
        
        if node.right:
            # Draw line to right child (most dissimilar)
            self.canvas.create_line(x, y+50, x_right, y_child-50, fill="red", width=2)
            self.canvas.create_text((x + x_right) // 2, (y+50 + y_child-50) // 2, 
                                  font=("Arial", 8, "bold"), fill="red")
            self._draw_tree(node.right, x_right, y_child, new_x_offset, depth + 1)


if __name__ == "__main__":
    image_paths = ["faces/" + f for f in os.listdir("faces") if f.endswith((".jpg", ".png"))]
    
    # Create the face graph
    print("Creating face similarity graph...")
    graph = FaceGraph(image_paths=image_paths, depth=3)
    
    # Create GUI and display
    root = tk.Tk()
    display = FaceGraphDisplay(root, graph)
    root.mainloop()