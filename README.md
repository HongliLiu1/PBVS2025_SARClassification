# 1st place solution for PBVS 2025 Multi-modal Aerial View Image Challenge - C (SAR Classification)
## Environment

Install the required dependencies by running `pip install -r requirements.txt` 
### Installation
## Preparation
- Download the dataset from [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/21244#participate-get_data) and put it in the `./datasets` folder.
## How To Test
Download the pretrained weights for the ResNet-50 and EfficientNet-B0 models, 
and save them in the appropriate directory. 
Ensure you update the paths in the code to point to the correct location of these weights.

For the dataset, place your test images in the specified folder (e.g., `img_folder = '/path_to_test_images'`). The images should be named according to the script's requirements.

Run the inference by executing `python test.py`. This will load the pretrained models, process the test images, generate predictions, and save the results in a `results.csv` file. The output CSV will contain columns for `image_id`, `class_id`, and `score`, representing the image ID, predicted class, and confidence score, respectively.

Once the script finishes running, submit the `results.csv` file to the competition platform.

If you are using a GPU, ensure that the device is correctly configured by setting `device_ids = [7]` and `device = f'cuda:{device_ids[0]}'`. Adjust the GPU or CPU settings based on your system.

Make# PBVS2025_SARClassification
# PBVS2025_SARClassification
# PBVS2025_SARClassification
