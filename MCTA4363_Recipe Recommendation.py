import torch
import cv2
from PIL import Image, ImageTk
from torchvision import models, transforms
from torchvision.transforms import functional as F
import tkinter as tk
from tkinter import ttk, filedialog
import webbrowser

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained object detection model
object_detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
object_detection_model.eval()
object_detection_model.to(device)

# Load the pre-trained image classification model
classification_model = torch.load('Ingredients.pt', map_location=device)
classification_model.eval()
classification_model.to(device)

# Define preprocessing transformations for classification model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust according to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the class names
class_names = ['Chicken', 'Eggs', 'Meat', 'Potato', 'Rice']  # Replace with your actual class names

# Dictionary of recipes based on combinations of ingredients and meal type
recipes = {
    frozenset(['Rice', 'Chicken']): {
        'Breakfast': [
            ("Chicken Congee", "https://christieathome.com/blog/chicken-congee/#recipe")
        ],
        'Lunch': [
            ("Chicken Rice", "https://farahjeats.com/recipe/malaysian-malay-style-chicken-rice-nasi-ayam/"),
        ],
        'Dinner': [
            (" Chicken fried rice ", "https://www.allrecipes.com/recipe/16954/chinese-chicken-fried-rice-ii")
        ]
    },
    frozenset(['Rice', 'Meat']): {
        'Breakfast': [
            ("Meat Gyudon", "https://farahjeats.com/recipe/gyudon-japanese-Meat-rice-bowl/")
        ],
        'Lunch': [
            ("Briyani Meat Rice", "https://www.recipesaresimple.com/recipe/nasi-briyani/")
        ],
        'Dinner': [
            ("Meat stir-fry with rice", "https://www.allrecipes.com/recipe/228823/quick-Meat-stir-fry")
        ]
    },
    frozenset(['Rice', 'Eggs']): {
        'Breakfast': [
            ("Fried Rice", "https://www.gimmesomeoven.com/fried-rice-recipe/")
        ],
        'Lunch': [
            ("Malaysian-Style Soy Sauce Eggs (Telur Masak Kicap)", "https://farahjeats.com/recipe/malaysian-soy-sauce-eggs-telur-masak-kicap/")
        ],
        'Dinner': [
            ("Nasi Goreng Pattaya", "https://www.unileverfoodsolutions.com.my/en/recipe/nasi-goreng-pattaya-R9002755.html")
        ]
    },
    frozenset(['Rice', 'Potato']): {
        'Breakfast': [
            ("Rice with Hash Brown", "https://www.spekkorice.co.za/recipes/rice-hash-browns-with-poached-eggs-tomato/")
        ],
        'Lunch': [
            ("Nasi Sambal Ikan Bilis with Potatoes", "https://friedchillies.com/fc/what-to-cook/malay-what-to-cook/sambal-ikan-bilis-with-potatoes/")
        ],
        'Dinner': [
            ("Mashed Potato Casserole With Rice", "https://recipes.net/main-dish/casserole/mashed-potatoes-and-rice-casserole-recipe/")
        ]
    },
    frozenset(['Chicken', 'Eggs']): {
        'Breakfast': [
            ("Chicken and eggs Wrap", "https://recipes.timesofindia.com/recipes/chicken-egg-wrap/rs75456053.cms")
        ],
        'Lunch': [
            ("Spicy chicken, egg & pepper Shakshuka", "https://www.mindfulchef.com/healthy-recipes/spicy-chicken-egg-and-pepper-shakshuka"),
        ],
        'Dinner': [
            ("Chicken Omelet", "https://www.thespruceeats.com/chicken-omelets-4589770")
        ]
    },
    frozenset(['Chicken', 'Potato']): {
        'Breakfast': [
            ("Garlic Butter Chicken with potato Skillet", "https://www.paleorunningmomma.com/garlic-butter-chicken-and-potato-skillet-whole30/")
        ],
        'Lunch': [
            ("Oyster Sauce Braised Chicken", "https://www.maggi.my/en/oyster-sauce-braised-chicken")
        ],
        'Dinner': [
            ("Roasted Chicken and Potatoes ", "https://www.jocooks.com/recipes/roast-chicken-with-roasted-potatoes/")
        ]
    },
    frozenset(['Meat', 'Eggs']): {
        'Breakfast': [
            ("Meat and eggs Burrito", "https://dudethatcookz.com/breakfast-cheesy-Meat-burrito/ ")
        ],
        'Lunch': [
            ("Egg Meat ", "https://tiffycooks.com/the-perfect-pairing-Meat-and-eggs-20-minutes/")
        ],
        'Dinner': [
            ("Steak & Eggs", "https://www.delish.com/cooking/recipe-ideas/a30433895/steak-and-eggs-recipe/")
        ]
    },
    frozenset(['Meat', 'Potato']): {
        'Breakfast': [
            ("Steak and Hash Brown", "https://www.tasteofhome.com/recipes/hash-brown-topped-steak/")
        ],
        'Lunch': [
            ("Meat and Potato Soy Sauce", "https://curatedkitchenware.com/blogs/soupeduprecipes/Meat-potatoes-over-rice")
        ],
        'Dinner': [
            ("Meat Stew with Carrots & Potatoes", "https://www.onceuponachef.com/recipes/Meat-stew-with-carrots-potatoes.html")
        ]
    },
    frozenset(['Eggs', 'Potato']): {
        'Breakfast': [
            ("Potato Frittata", "https://www.allrecipes.com/recipe/47044/spinach-and-potato-frittata/")
        ],
        'Lunch': [
            ("Egg and Potato Curry", "https://www.yeahfoodie.com/main-course/simple-malaysian-egg-and-potato-curry-recipe/")
        ],
        'Dinner': [
            ("Skillet Potato and Egg", "https://www.aberdeenskitchen.com/2018/02/skillet-potato-and-egg-hash/")
        ]
    },
    frozenset(['Rice', 'Chicken', 'Eggs']): {
        'Breakfast': [
            ("Nasi Lemak Ayam", "https://makan.ch/recipe/nasi-lemak/")
        ],
        'Lunch': [
            ("Chicken Katsudon (Chicken Cutlet Rice Bowl)", "https://farahjeats.com/recipe/chicken-katsudon-chicken-cutlet-rice-bowl/")
        ],
        'Dinner': [
            ("Chicken Katsudon", "https://farahjeats.com/recipe/chicken-katsudon-chicken-cutlet-rice-bowl/")
        ]
    },
    frozenset(['Rice', 'Chicken', 'Potato']): {
        'Breakfast': [
            ("Chicken Curry with rice", "https://www.chelseasmessyapron.com/easy-chicken-curry-and-rice-one-skillet/")
        ],
        'Lunch': [
            ("Japanese Chicken Katsu Curry", "https://farahjeats.com/recipe/japanese-chicken-katsu-curry/")
        ],
        'Dinner': [
            ("Chicken and Potato Casserole", "https://www.allrecipes.com/recipe/279847/chicken-and-potato-casserole/")
        ]
    },
    frozenset(['Rice', 'Meat', 'Eggs']): {
        'Breakfast': [
            ("Bulgogi Kimchi fried Rice", "https://moribyan.com/bulgogi-kimchi-fried-rice/")
        ],
        'Lunch': [
            ("Hawaiian Loco Moco", "https://farahjeats.com/recipe/hawaiian-loco-moco/")
        ],
        'Dinner': [
            ("Meat bibimbap", "https://www.havehalalwilltravel.com/easy-bibimbap-recipe-halal")
        ]
    },
    frozenset(['Rice', 'Meat', 'Potato']): {
        'Breakfast': [
            ("Meat Empanada Rice", "https://salimaskitchen.com/one-pot-empanada-rice/")
        ],
        'Lunch': [
            ("Spicy Fried Meat", "https://farahjeats.com/recipe/spicy-fried-Meat-daging-goreng-berlada/")
        ],
        'Dinner': [
            ("Meat and egg stir fry rice bowls ", "https://thewoksoflife.com/Meat-egg-stir-fry-rice/")
        ]
    },
    frozenset(['Meat', 'Eggs', 'Potato']): {
        'Breakfast': [
            ("Meat Quesadilla with eggs", "https://streetsmartnutrition.com/breakfast-Meat-quesadillas-with-steak-and-eggs/")
        ],
        'Lunch': [
            ("Mayak Eggs (Korean Marinated Eggs)", "https://farahjeats.com/recipe/mayak-eggs-korean-marinated-eggs/")
        ],
        'Dinner': [
            ("Steak Potato and Egg Hash", "https://whatmollymade.com/steak-and-potato-breakfast-hash/")
        ]
    },
    frozenset(['Rice', 'Eggs', 'Potato']): {
        'Breakfast': [
            ("Sushi Bowl with spicy Mayo", "https://ourbalancedbowl.com/homemade-sushi-bowls-with-spicy-mayo/")
        ],
        'Lunch': [
            ("Corned Meat Potatoes and Eggs", "https://mayakitchenette.com/corned-Meat-potatoes-eggs/")
        ],
        'Dinner': [
            ("Spanish Rice with Egg", "https://cookaifood.com/recipe/spanish-rice-with-egg")
        ]
    },
}

# Function to classify an ingredient
def classify_ingredient(crop):
    # Preprocess the cropped image
    image_tensor = preprocess(crop)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)  # Move tensor to the correct device

    # Perform inference
    with torch.no_grad():
        outputs = classification_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class

# Function to detect ingredients and display image
def process_image(image_path):
    detected_ingredients = set()

    # Load and prepare the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Detect objects in the image
    with torch.no_grad():
        detections = object_detection_model(image_tensor)

    # Convert detections to CPU for further processing
    detections = [{k: v.to('cpu') for k, v in t.items()} for t in detections]

    # Load the image using OpenCV for drawing
    image_cv = cv2.imread(image_path)
    height, width, _ = image_cv.shape

    # Loop through detected objects
    for i in range(len(detections[0]['boxes'])):
        box = detections[0]['boxes'][i].numpy().astype(int)
        score = detections[0]['scores'][i].item()
        
        # Only consider detections with a confidence score above a threshold (e.g., 0.5)
        if score > 0.5:
            x1, y1, x2, y2 = box
            crop = image.crop((x1, y1, x2, y2))
            predicted_class = classify_ingredient(crop)
            detected_ingredients.add(predicted_class)

            # Draw the bounding box and label on the image
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_cv, predicted_class, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Resize the image for display purposes
    max_height = 800  # You can adjust this value
    scale = max_height / height
    new_width = int(width * scale)
    resized_image = cv2.resize(image_cv, (new_width, max_height))

    # Convert OpenCV image to PIL format for tkinter
    image_pil = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    # Display the image using tkinter
    img_tk = ImageTk.PhotoImage(image_pil)
    img_label.config(image=img_tk)
    img_label.image = img_tk  # Keep a reference to avoid garbage collection

    # Ask the user to select a meal type
    select_meal_type(detected_ingredients)

def select_meal_type(detected_ingredients):
    meal_type_window = tk.Toplevel(root)
    meal_type_window.title("Select Meal Type")
    meal_type_window.configure(bg='pink')

    meal_type_label = tk.Label(meal_type_window, text="Select Meal Type:", font=("Helvetica", 16), bg='pink')
    meal_type_label.pack(pady=10)

    meal_type_var = tk.StringVar(value="Breakfast")

    breakfast_radio = tk.Radiobutton(meal_type_window, text="Breakfast", variable=meal_type_var, value="Breakfast", font=("Helvetica", 14), bg='pink')
    breakfast_radio.pack(pady=5)
    
    lunch_radio = tk.Radiobutton(meal_type_window, text="Lunch", variable=meal_type_var, value="Lunch", font=("Helvetica", 14), bg='pink')
    lunch_radio.pack(pady=5)
    
    dinner_radio = tk.Radiobutton(meal_type_window, text="Dinner", variable=meal_type_var, value="Dinner", font=("Helvetica", 14), bg='pink')
    dinner_radio.pack(pady=5)

    select_button = tk.Button(meal_type_window, text="Select", command=lambda: display_recipes(detected_ingredients, meal_type_var.get(), meal_type_window), font=("Helvetica", 14), bg='pink')
    select_button.pack(pady=10)

def display_recipes(detected_ingredients, meal_type, meal_type_window):
    meal_type_window.destroy()

    text.delete('1.0', tk.END)
    text.insert(tk.END, f"Recipe Recommendations for {meal_type}:\n")

    # Check for matching recipe combinations
    found_recipes = False
    for ingredients, meal_types in recipes.items():
        if ingredients.issubset(detected_ingredients):
            if meal_type in meal_types:
                found_recipes = True
                # Insert the text with red color
                text.insert(tk.END, f"\nRecipes with {', '.join(ingredients)}:\n", "red")
                text.tag_configure("red", foreground="red")  # Configure tag for red color
                for recipe_name, recipe_url in meal_types[meal_type]:
                    text.insert(tk.END, f"{recipe_name}: {recipe_url}\n")
                    text.tag_configure("link", foreground="blue", underline=True)
                    text.tag_bind("link", "<Button-1>", lambda e, url=recipe_url: open_recipe(url))

    if not found_recipes:
        text.insert(tk.END, f"\nNo {meal_type} recipes found for the detected ingredients.\n")

    # Show the buttons on the second page
    finish_button.pack(side=tk.RIGHT, padx=5, pady=10)

# Function to open a file dialog and select an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
    if file_path:
        process_image(file_path)

# Function to open a URL in the default web browser
def open_recipe(url):
    webbrowser.open(url)

# Function to close the application
def finish():
    root.destroy()

# Function to reset the application to upload another image
def upload_another():
    text.delete('1.0', tk.END)
    img_label.config(image='')
    img_label.image = None
    finish_button.pack_forget()

# Create the main application window
root = tk.Tk()
root.title("Ingredient Classification and Recipe Recommendations")
root.configure(bg='pink')

# Add title to the application window
title_label = tk.Label(root, text="What's on the Menu Today?", font=("Comic Sans MS", 28, "bold"), bg='pink')
title_label.pack(pady=20)

# Create and pack the image frame
img_frame = tk.Frame(root, bg='pink')
img_frame.pack(side=tk.LEFT, padx=10, pady=10)
img_label = tk.Label(img_frame, bg='pink')
img_label.pack()

# Create and pack the recipe frame
recipe_frame = tk.Frame(root, bg='pink')
recipe_frame.pack(side=tk.RIGHT, padx=10, pady=10)
text = tk.Text(recipe_frame, wrap=tk.WORD, width=40, height=40, bg='white')
text.pack()

# Create and pack the upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image, bg='white')
upload_button.pack(side=tk.TOP, pady=10)

# Create buttons for uploading another image and finishing
buttons_frame = tk.Frame(root, bg='pink')
buttons_frame.pack(side=tk.BOTTOM, pady=10)
finish_button = tk.Button(buttons_frame, text="Let's Cook", command=finish, bg='white')

root.mainloop()