# Summary
Personal POC to use YOLO11n (Ultralytics Python library) to train an AI model to visually detect webpage buttons and input fields in order to integrate with Selium.

# Results
## Labeling
Label Studio was used to upload a custom dataset of 67 screenshots. Each screenshot was annotated with either 'button' or 'input'. This annotated dataset was then exported in YOLO format and added to this project.


# Libraries
pip install ultralytics
pip install easyocr

#Labeling Definitions
- Buttons: Always have text (icons can be buttons but don't have to). If icons are defined as buttons, then icons that are not buttons will be found as buttons.
- Inputs: A field where data can be entered/updated/viewed. Labels should be labelled and a relationship created from the input boundary box to the label. A label that is an icon should also be a 'label' (e.g., a magnifying glass for search bar).

#Folder Structure
> datasets: Directory that holds /images/ and /labels/ (labelled images used to train an AI model)
> input: validation images to output detection scores (don't think this is neede because .yaml has /images/ for 'vals')
> predict: Used to detect objects in images to test the model. Before running predict.py, you need to copy the best.pt and a screenshot into this directory.

# Setting up Label Studio
- `pip install -U label-studio`
- `export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`
- `label-studio`
- Open `localhost:8080` and login (if need a login, you can create a free account)
- Create or use existing project for object detection using boundary boxes
- Label all screenshots
- In top-right corner, export to `Yolo`
- Unzip file in `Downloads`
- Put all contents in ./Datasets directory

# Train model
- In Terminal, `cd ./computer_vision_ai_training`
- `python3 ./training_script.py`
- When training finishes, copy best.pt to ./predict directory
- `python3 ./predict.py`

# OCR AI FYI
Download `craft_mlt_25k.pth` and `english_g2.pth` to `./ocr/model` in order to run OCR in the predict script. They are too large to upload into github. They can be found on EasyOCR's website.

Journal:
- Day 1: 
    - yolo11n works great, with 53 images and sloppy labelling, I was able to train a model in .6hrs and it detects objects... BUT the objects are reversed. So it's finding inputs as buttons and buttons as inputs. Need to see if I messed something up in labeling. It's the classes yaml file - I had the classes reversed.
    - Retrained with reversed yaml vals. output in 'train' directory. To note, Label Studio export classes.txt should mirror the order in data.yaml.
- Day 2: 
    - Tightened bounding boxes around buttons and inputs. Added labels as category for inputs. Removed all buttons unless they had text+icon, or icon w/pill or text w/pill. Model found both inputs w/labels, but only one login button... the sso button. It really needs to find the main login button. I added ~10 login screenshots and labelled them. Found in train3.
    - Results are better but didn't find a label when resizing prediction image to 640 (default). I can live with this. Figured out to how to crop each finding to memory (numpy array) and use easyocr to grab the text, but it keeps mispelling "Username" to "Usernare". Need to consider image classification. When I predicted at 800,1280 - got totally different results. Next, I need to try and train a model with higher resolution dataset. Consider 600,1000, but might need to stream it in to avoid memory issues by using cache=True. Apparently cache=true can lead to non-deterministic results. Try cache='disk'. Also need toupdate  to 1024.
    - 2hrs to train model with image size of 1024. Great results, however it's finding a 4th button (by combining two buttons). The ocr confidene is 54-57%, so when running ocr, I throw out anything below a 65%. So for my purposes, a button with no text  will be thrown out.