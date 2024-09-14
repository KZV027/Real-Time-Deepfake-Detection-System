import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = r'C:\Users\kshit\best_deepfake_model.h5'  # Use raw string to handle backslashes in path
model = tf.keras.models.load_model(model_path)

# Preprocess the image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Adjust size as needed for the model
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Classify the prediction result
def classify_prediction(prediction):
    probability = tf.sigmoid(prediction[0][0]).numpy()  # Apply sigmoid to the prediction
    threshold = 0.51  # Set threshold for classification
    return "Deepfake" if probability < threshold else "Real"

# Update the interface with the prediction result
def update_interface_with_result(image, classification):
    # Convert the OpenCV image to PIL format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Resize the image to fit the display size
    display_width, display_height = 400, 300  # Adjust size as needed
    image = image.resize((display_width, display_height), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    # Update the image and result on the GUI
    image_label.config(image=photo, width=display_width, height=display_height)
    image_label.image = photo
    result_label.config(text=f"Classification: {classification}")

# Handle image upload and prediction
def upload_image():
    stop_webcam()  # Ensure the webcam is turned off when uploading an image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)  # Read the selected image
        preprocessed_image = preprocess_image(image)  # Preprocess the image
        prediction = model.predict(preprocessed_image)  # Predict using the model
        classification = classify_prediction(prediction)  # Classify the prediction
        update_interface_with_result(image, classification)  # Update GUI with the results

# Stop the webcam feed
def stop_webcam():
    if hasattr(root, 'webcam_app') and root.webcam_app.cap.isOpened():
        root.webcam_app.cap.release()
        root.unbind('<Return>')  # Unbind the Enter key to stop capturing
        root.unbind('q')  # Unbind the q key
        if root.webcam_app.webcam_label.image:
            root.webcam_app.webcam_label.config(image='', text="Webcam feed stopped")  # Clear the feed

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.cap = cv2.VideoCapture(0)
        self.frame = None

        # Create a label to display real-time webcam feed
        self.webcam_label = tk.Label(root, text="Webcam feed will show here", font=("Helvetica", 14), width=400, height=300, borderwidth=2, relief="ridge", bg='#444444', fg='white')
        self.webcam_label.pack(pady=20)

        # Bind keys for capturing and quitting
        self.root.bind('<Return>', self.capture_image)
        self.root.bind('q', self.quit_application)

        # Start updating the webcam feed in the label
        self.update_webcam_feed()

    def update_webcam_feed(self):
        if not self.cap.isOpened():
            return

        ret, self.frame = self.cap.read()
        if ret:
            # Convert the frame to PIL format and update the label
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # Resize the frame to a smaller, fixed display size
            display_width, display_height = 400, 300  # Adjust as needed
            frame_pil = frame_pil.resize((display_width, display_height), Image.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.webcam_label.config(image=frame_tk, text='')  # Clear text when displaying the image
            self.webcam_label.image = frame_tk
        self.webcam_label.after(10, self.update_webcam_feed)  # Update every 10 ms

    def capture_image(self, event):
        if self.frame is not None:
            preprocessed_frame = preprocess_image(self.frame)  # Preprocess the captured frame
            prediction = model.predict(preprocessed_frame)  # Predict using the model
            classification = classify_prediction(prediction)  # Classify the prediction
            update_interface_with_result(self.frame, classification)  # Update GUI with the results

    def quit_application(self, event):
        self.cap.release()
        root.quit()

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Deepfake Detection")
root.geometry("1000x800")  # Adjust the window size for better display
root.configure(bg='#1e1e1e')  # Set the entire window background to dark

# Function to animate button on hover
def on_enter(e):
    e.widget['background'] = '#444444'  # Darker grey on hover

def on_leave(e):
    e.widget['background'] = '#2b2b2b'  # New original color when not hovered

def on_click(e):
    e.widget['background'] = '#333333'  # Even darker color when clicked

def on_release(e):
    e.widget['background'] = '#444444'  # Return to hover color

# Heading Label
heading_label = tk.Label(root, text="Real-Time Deepfake Detection Using Deep Learning", font=("Helvetica", 18, "bold"), bg='#2e2e2e', fg='white')
heading_label.pack(pady=20)
# Frame for buttons
button_frame = tk.Frame(root, bg='#1e1e1e')  # Ensures frame background matches the dark theme
button_frame.pack(pady=20)

# Upload button with hover and click animations
upload_btn = tk.Button(
    button_frame, text="Upload Image", command=upload_image,
    bg='#2b2b2b', fg='white', font=("Helvetica", 12, "bold"),
    padx=20, pady=10, activebackground='#333333', activeforeground='white',
    relief='flat', bd=2, cursor='hand2'
)
upload_btn.bind("<Enter>", on_enter)
upload_btn.bind("<Leave>", on_leave)
upload_btn.bind("<ButtonPress-1>", on_click)
upload_btn.bind("<ButtonRelease-1>", on_release)
upload_btn.grid(row=0, column=0, padx=10)

# Webcam button with hover and click animations
webcam_btn = tk.Button(
    button_frame, text="Webcam Detection", command=lambda: start_webcam(),
    bg='#2b2b2b', fg='white', font=("Helvetica", 12, "bold"),
    padx=20, pady=10, activebackground='#333333', activeforeground='white',
    relief='flat', bd=2, cursor='hand2'
)
webcam_btn.bind("<Enter>", on_enter)
webcam_btn.bind("<Leave>", on_leave)
webcam_btn.bind("<ButtonPress-1>", on_click)
webcam_btn.bind("<ButtonRelease-1>", on_release)
webcam_btn.grid(row=0, column=1, padx=10)

# Label to display the image with resized dimensions and border
image_label = tk.Label(
    root, text="Your image will show here", font=("Helvetica", 14, "italic"),
    width=50, height=15, borderwidth=5, relief="ridge",  # Adjusted width and height
    bg='#444444', fg='white'
)
image_label.pack(pady=20)

# Result label for displaying detection results
result_label = tk.Label(
    root, text="", font=("Helvetica", 18, "bold"),
    bg='#1e1e1e', fg='yellow'
)
result_label.pack(pady=20)

def start_webcam():
    stop_webcam()  # Stop any existing webcam feeds before starting a new one
    root.webcam_app = WebcamApp(root)  # Assign the webcam app to root to manage its state

# Start the Tkinter main loop
root.mainloop()
