# Declaration
All code in `CAPTCHA/solving_captchas_code_examples` is provided by Adam Geitgey on [How to break a CAPTCHA system in 15 minutes with Machine Learning](https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710). I just made some limited modifications.

# 声明
本项目中`CAPTCHA/solving_captchas_code_examples` 目录中的所有代码实现均由Adam Geitgey在[How to break a CAPTCHA system in 15 minutes with Machine Learning](https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710)上提供. 本人在此基础上做了有限的、适当的修改和完善.

### Before you get started

To run these scripts, you need the following installed:

1. Python 3
2. OpenCV 3 w/ Python extensions
 - I highly recommend these OpenCV installation guides: 
   https://www.pyimagesearch.com/opencv-tutorials-resources-guides/ 
3. The python libraries listed in requirements.txt
 - Try running "pip3 install -r requirements.txt"

### Step 1: Extract single letters from CAPTCHA images

Run:

python3 extract_single_letters_from_captchas.py

The results will be stored in the "extracted_letter_images" folder.


### Step 2: Train the neural network to recognize single letters

Run:

python3 train_model.py

This will write out "captcha_model.hdf5" and "model_labels.dat"


### Step 3: Use the model to solve CAPTCHAs!

Run: 

python3 solve_captchas_with_model.py