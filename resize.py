import os
import cv2

# Specify the input and output directories
input_directory = 'x'
output_directory = 'BanglaHandwrittenCharacterDataset\Train'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over the folders in the input directory
for folder_name in os.listdir(input_directory):
    folder_path = os.path.join(input_directory, folder_name)
    print(folder_name)
    # Check if the current item is a directory
    if os.path.isdir(folder_path):
        # Create the corresponding output folder
        output_folder = os.path.join(output_directory, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate over the images in the current folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Load the image using OpenCV
            image = cv2.imread(image_path)
            print(image_name)
            # Resize the image to 32x32
            if image_name == "desktop.ini":
                continue
            resized_image = cv2.resize(image, (32, 32),interpolation=cv2.INTER_AREA)

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, resized_image)

print("Image conversion and saving complete.")
