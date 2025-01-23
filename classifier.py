# Import required libraries
import pygame          # GUI and drawing
import numpy as np     # Numerical operations
from PIL import Image  # Image processing
from sklearn.ensemble import RandomForestClassifier  # ML classifier
from sklearn.datasets import load_digits  # Digit dataset
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.metrics import ConfusionMatrixDisplay  # Evaluation visualization
import matplotlib.pyplot as plt  # Plotting

# Load and preprocess dataset
digits = load_digits()
X = digits.data
y = digits.target
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scale features

# Create and train improved model
model = RandomForestClassifier(n_estimators=200, max_depth=20)
model.fit(X, y)

# # Display model performance
# ConfusionMatrixDisplay.from_estimator(model, X, y)
# plt.show()

# Initialize pygame window
pygame.init()
screen = pygame.display.set_mode((280, 280))  # Window size
pygame.display.set_caption('Draw a digit')
drawing = False
screen.fill((255, 255, 255))  # White background
pygame.display.flip()

def process_screen():
   # Convert pygame surface to image
   strFormat = 'RGB'
   raw_str = pygame.image.tostring(screen, strFormat, False)
   image = Image.frombytes(strFormat, screen.get_size(), raw_str)
   image = image.convert('L')  # Convert to grayscale
   
   # Process image for better recognition
   image = Image.fromarray(255 - np.array(image))  # Invert colors
   image = image.point(lambda x: 0 if x < 128 else 255)  # Enhance contrast
   image = image.resize((8, 8), Image.Resampling.LANCZOS)  # Resize with better algorithm
   
   # Show processed image
   plt.figure(figsize=(2, 2))
   plt.imshow(np.array(image), cmap='gray')
   plt.title('Processed Image')
   plt.axis('off')
   plt.show()
   
   # Prepare image for model
   img_array = np.array(image).flatten() / 16.0  # Normalize
   img_array = scaler.transform([img_array])  # Scale like training data
   return img_array

# Main program loop
running = True
while running:
   for event in pygame.event.get():
       if event.type == pygame.QUIT:  # Close window
           running = False
       elif event.type == pygame.MOUSEBUTTONDOWN:  # Start drawing
           drawing = True
           print("Started drawing")
       elif event.type == pygame.MOUSEBUTTONUP:  # Stop drawing
           drawing = False
           print("Stopped drawing")
       elif event.type == pygame.MOUSEMOTION and drawing:  # Draw while moving
           pygame.draw.circle(screen, (0, 0, 0), event.pos, 20)  # Thick black line
       elif event.type == pygame.KEYDOWN:
           if event.key == pygame.K_RETURN:  # Make prediction
               print("Processing...")
               digit_array = process_screen()
               prediction = model.predict(digit_array)
               confidence = model.predict_proba(digit_array).max()
               print(f"Predicted digit: {prediction[0]} (Confidence: {confidence:.2%})")
           elif event.key == pygame.K_c:  # Clear screen
               print("Clearing screen")
               screen.fill((255, 255, 255))
       
       pygame.display.flip()  # Update display

pygame.quit()  # Cleanup