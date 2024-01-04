#Input: URL
#Output: Positions of Elements 
#One Script to get Coordinates of Some Buttons.
import numpy as np
import pyautogui
import time 
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import torch 
import torch.nn.functional as F 
from torchvision import transforms 
from PIL import Image, ImageDraw 
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import 
#mobile_emulation = {
 #  "deviceMetrics": {"width": 1080, "height": 1920, "pixelRatio": 3.0},
  #  "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1",
#}

chrome_options = webdriver.ChromeOptions()
#chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)

# Start the Chrome WebDriver with the specified options
driver = webdriver.Chrome(options=chrome_options)
print("driver set up ")
# Navigate to the URL
#baseUrl = "https://transportplanner.azurewebsites.net/Home?ReturnUrl=%2fHome%2fHome"
baseUrl = "https://practicetestautomation.com/practice-test-login/"
driver.get(baseUrl)
driver.maximize_window()
time.sleep(20)

path = "C:/Users/Public/Documents/AIScreenshots/A.png"
driver.save_screenshot(path)
print("Screenshot Saved")
test_image = Image.open(path)


#PNG files are much better when converting to RGB, retains quality
test_image = test_image.convert("RGB")
rgb_path = "C:/Users/Public/Documents/AIScreenshots/RGB.png"
test_image.save(rgb_path)


print("Converted to RGB ! ")

print("Loading Screen Recognition Model")
UIElementDetection_Time = time.time()
m = torch.jit.load('C:/Users/Public/Documents/AIScreenshots/Torchscripts/screenrecognition-web7k-vins.torchscript', map_location=torch.device('cpu'))
class_map_file = "C:/Users/Public/Documents/AIScreenshots/Torchscripts/class_map_vins_manual.json"
with open(class_map_file, "r") as f:
    class_map = json.load(f)
idx2Label = class_map['idx2Label']
img_transforms = transforms.ToTensor()
img_input = img_transforms(test_image)
pred = m([img_input])[1]



import torch

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU for computations.")
else:
    print("CUDA is not available. GPU support is not enabled.")


import os
draw = ImageDraw.Draw(test_image)
output_path = "C:/Users/Public/Documents/AIScreenshots/OutputImages/"
  # Replace this with your desired output path

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)
conf_thresh = 0.4
for i in range(len(pred[0]['boxes'])):
    conf_score = pred[0]['scores'][i]
    if conf_score > conf_thresh:
        x1, y1, x2, y2 = pred[0]['boxes'][i]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        draw.rectangle([x1, y1, x2, y2], outline='white')
        #draw.text((x1, y1), idx2Label[str(int(pred[0]['labels'][i]))] + " {:.2f}".format(float(conf_score)), fill="red")
        box_image = test_image.crop((x1, y1, x2, y2))
        # Save the extracted bounding box image
        
        #PNG --> BETTER FILE FORMAT
        image_path = os.path.join(output_path, f"box_{i}.png")
        box_image.save(image_path)
        test_image.save(rgb_path)
UIElementDetection_EndTime = time.time() - UIElementDetection_Time 
print("UI Detection time: ", UIElementDetection_EndTime)
driver.quit()



print("Text Identification Starts")
import keras_ocr
import matplotlib.pyplot as plt
ocr_start_time = time.time()
pipeline = keras_ocr.pipeline.Pipeline()


import os
import keras_ocr.tools

# Path to the folder containing images in your Google Drive


# List all the files in the images_folder_path
file_list = os.listdir(output_path)

# Filter out only image files (you can add more file extensions as needed)
image_files = [file for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Read all the images and store them in a list
images = [
    keras_ocr.tools.read(os.path.join(output_path, img_file)) for img_file in image_files
]

prediction_groups = pipeline.recognize(images)
print("Number of elements detected: ", len(prediction_groups))
base_filename = 'annotated_image'
counter = 1

for image, predictions in zip(images, prediction_groups):
    # Create a copy of the image as a NumPy array
    image_with_annotations = np.array(image.copy())

    # Draw the annotations on the image
    keras_ocr.tools.drawAnnotations(image=image_with_annotations,
                                    predictions=predictions)

    # Convert the NumPy array to a PIL Image
    image_with_annotations_pil = Image.fromarray(image_with_annotations)

    # Generate a unique filename for each annotated image
    unique_filename = f'{base_filename}_{counter}.png'

    # Save the image with annotations using the unique filename
    image_with_annotations_pil.save(unique_filename)

    # Increment the counter for the next image
    counter += 1
ocr_end_time = time.time() - ocr_start_time
print("OCR timing: ", ocr_end_time) 

def getElementFromCoordinates(x,y):
    x_coord = x  
    y_coord = y 
    script = f'''
        var element = document.elementFromPoint({x_coord}, {y_coord});
        return element;
    '''
    element = driver2.execute_script(script)
    return element

def SendData(theElement, payload):
    driver2.execute_script("arguments[0].value = arguments[1];", theElement, payload)



usernameCount  = 0 
for i in prediction_groups:
  for text, box in i:
    if(text=='Submit'):
       print('Username text has been detected:', text, usernameCount)
       break
   
  if(text=='Submit'):
      
      break
  else :
      usernameCount = usernameCount + 1
print('Username Count: ', usernameCount)

passwordCount  = 0 
for i in prediction_groups:
  for text, box in i:
    if(text=='password'):
       print('Password text has been detected:', text, passwordCount)
       break
  
  if(text=='password'):
      
      break
  else:
      passwordCount = passwordCount + 1 
  
print('Password Count: ',  passwordCount)


loginCount = 0
for i in prediction_groups:
  for text, box in i:
    if(text=='login'):
       print('Login text has been detected:', text, loginCount)
       break
  
  if(text=='login'):
      
      break
  else:
      loginCount = loginCount + 1
  
print('Login Count: ',  loginCount)



print("Coordinates Finding Begin!")
import cv2
import numpy as np

def find_snippet_location(full_image_path, snippet_image_path):
    full_image = cv2.imread(full_image_path)
    snippet_image = cv2.imread(snippet_image_path)

    result = cv2.matchTemplate(full_image, snippet_image, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    snippet_height, snippet_width, _ = snippet_image.shape
    x, y = max_loc

    # Calculate the midpoint of the snippet
    midpoint_x = x + snippet_width // 2
    midpoint_y = y + snippet_height // 2

    return midpoint_x, midpoint_y





full_image_path = rgb_path
file_paths = [] 
for filename in os.listdir(output_path):
    file_path = os.path.join(output_path, filename)
    if os.path.isfile(file_path):
        file_paths.append(file_path)

 # Replace with the path to the full image
for snippet_image_path in file_paths:   # Replace with the path to the snippet image

    midpoint_x, midpoint_y = find_snippet_location(full_image_path, snippet_image_path)
    midpoint_y = midpoint_y 
    print(f"Snippet midpoint (x, y): ({midpoint_x}, {midpoint_y})")




#ScreenCalibration 

print("Gazing for Username... ")
#USERNAME
 
username_box = f"box_{usernameCount}.png"
username_pos = output_path + username_box 
print(username_pos)
midpoint_x, midpoint_y = find_snippet_location(full_image_path, username_pos)
x_coordinate = midpoint_x 
y_coordinate = midpoint_y 
driver2 = webdriver.Chrome(options=chrome_options)
driver2.get(baseUrl)
driver2.maximize_window()
time.sleep(20)
print(x_coordinate, y_coordinate)
username = getElementFromCoordinates(x_coordinate, y_coordinate)
username.click()
SendData(username, '315750')

print("Gazing for password...")
#PASSWORD

password_box = f"box_{passwordCount}.png"
password_pos = output_path + password_box 
print(password_pos)
midpoint_x, midpoint_y = find_snippet_location(full_image_path, password_pos)
x_coordinate = midpoint_x 
y_coordinate = midpoint_y 
print(x_coordinate, y_coordinate)
password = getElementFromCoordinates(x_coordinate, y_coordinate)
password.click()
SendData(password, '315750')


print("Gazing for Login...")
#LOGIN

box_name = f"box_{loginCount}.png"
login_pos = output_path + box_name 
print(login_pos)
midpoint_x, midpoint_y = find_snippet_location(full_image_path, login_pos)
x_coordinate = midpoint_x 
y_coordinate = midpoint_y 
print(x_coordinate, y_coordinate)
loginBtn = getElementFromCoordinates(x_coordinate, y_coordinate)
loginBtn.click()


time.sleep(10)     
driver.quit()




 

