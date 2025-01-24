from ultralytics import YOLO
import easyocr
import cv2 
import numpy as np

# Ultralytics documentation: https://docs.ultralytics.com/modes/predict/#working-with-results
# EasyOCR documentation: https://www.jaided.ai/easyocr/documentation/
reader = easyocr.Reader(lang_list=['en'], 
                        model_storage_directory='./computer_vision_ai_training/ocr/model') # this needs to run only once to load the model into memory

def crop_box(image, obb):
    """ Crop the Oriented Bounding Box (OBB) """
    x, y, w, h = map(int, obb)

    rot_rect = ((x, y), (w, h), 0)

    # Get the rotation matrix
    box = cv2.boxPoints(rot_rect)
    box = np.intp(box)

    # Crop the bounding box region
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, h-1],
                        [0, 0],
                        [w-1, 0],
                        [w-1, h-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_img = cv2.warpPerspective(image, M, (w, h))
    
    return cropped_img

# Load a model
model = YOLO("./computer_vision_ai_training/predict/best.pt")  # load a custom model

# Predict with the model
results = model("./computer_vision_ai_training/predict/login.png", imgsz=1024, line_width=1, classes=[0,1], augment=True)  # predict on an image

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs

    for box in boxes:
        x, y, width, height = map(int, box.xywh[0])  # Convert coordinates to integers
        class_id = int(box.cls.item())  # Convert class ID to integer
        class_name = result.names[class_id]  # Get the class name from the result.names
        crop_img = crop_box(result.orig_img, box.xywh[0])
        
        # # Perform OCR only if the cropped image is not empty
        # if crop_img.size > 0:
        #     ocr_out = reader.readtext(crop_img, adjust_contrast=0.1, contrast_ths=0.7) # Contract .7 or lower gets a second pass with contrast of .1

        #     if ocr_out:
        #         text = ocr_out[0][1]
        #         conf = ocr_out[0][2]

        #         if conf < 0.65:
        #             text = ""
        #     else:
        #         text = ""
        #         conf = 0.0
        # else:
        #     text = ""
        #     conf = 0.0
        
        #print(f"class={class_name}, x={x}, y={y}, width={width}, height={height}, confidence={box.conf.item():.2f}, text={text}, ocr_confidence={conf*100:.0f}%")
        print(f"class={class_name}, x={x}, y={y}, width={width}, height={height}, confidence={box.conf.item():.2f}")

        # Save the cropped image
        #save_path = f"/Users/paul.felten/bot_party_training/predict/test4/cropped_{class_name}_{x}_{y}.png"
        #cv2.imwrite(save_path, crop_img)
        
    result.show()  # display to screen
    result.save(filename="./computer_vision_ai_training/predict/result.jpg")  # save to disk

