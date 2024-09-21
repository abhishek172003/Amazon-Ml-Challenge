import os
import re
import requests
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError
import pytesseract
import pandas as pd
from sympy import evaluate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.constants import allowed_units, unit_full_name_map, entity_unit_map

# Preprocess image only when needed (based on image quality)
def preprocess_image(image, enhance=False):
    image = image.convert('L')
    if enhance:
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(2.0)
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(2.0)
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return image

# Function to download image content as bytes
def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.RequestException as e:
        print(f"Error downloading image {image_url}: {e}")
        return None

# Function to extract text from an image file
def extract_text_from_image(image_bytes, enhance=False):
    try:
        image = Image.open(image_bytes)
        image = preprocess_image(image, enhance=enhance)
        config = "--psm 6"
        text = pytesseract.image_to_string(image, config=config)
        return text
    except Exception as e:
        print(f"An error occurred while extracting text: {e}")
        return None

# Function to generate a regex pattern based on the entity's units
def get_unit_pattern(units):
    unit_regex = r'\b(\d+(\.\d+)?)\s*({})\b'.format('|'.join(units))
    return re.compile(unit_regex, re.IGNORECASE)

# Function to extract the highest value unit from text based on entity's units
def extract_highest_unit(entity_name, text):
    if entity_name not in entity_unit_map:
        return None

    # Get the set of valid units for the entity
    valid_units = entity_unit_map[entity_name]
    
    # Generate regex pattern for short form units
    unit_pattern = get_unit_pattern(unit_full_name_map.keys())
    
    # Find all matches of value + unit
    matches = unit_pattern.findall(text)
    if not matches:
        return None
    
    # Process each match: convert short unit to full form and check validity
    valid_matches = []
    for match in matches:
        value = float(match[0])
        short_unit = match[2].lower()
        
        # Convert short form to full form
        full_unit = unit_full_name_map.get(short_unit, short_unit)
        
        # Only keep valid units for the entity
        if full_unit in valid_units:
            valid_matches.append((value, full_unit))
    
    if not valid_matches:
        return None

    # Find the highest value
    max_value, max_unit = max(valid_matches, key=lambda x: x[0])

    # Filter the results based on allowed units
    if max_unit not in allowed_units:
        return None  # Skip if the max unit is not allowed

    # Format max_value: add decimal if float, leave as is if integer
    if max_value.is_integer():
        formatted_value = f"{int(max_value)}"  # No decimal for integers
    else:
        formatted_value = f"{max_value:.2f}"  # Two decimal places for floats

    return f"{formatted_value} {max_unit}"

# The predictor function integrated with image processing and text extraction
def predictor(image_link, group_id, entity_name):
    # Download the image content as bytes
    image_bytes = download_image(image_link)
    if not image_bytes:
        return {"index": group_id, "prediction": None}

    # Extract text from the image
    text = extract_text_from_image(image_bytes, enhance=False)
    if text is None or text.strip() == '':
        text = extract_text_from_image(image_bytes, enhance=True)
    
    # Extract the highest value unit from the text using entity name
    highest_unit = extract_highest_unit(entity_name, text)
    
    return {"index": group_id, "prediction": highest_unit}

# Function to evaluate the model
def evaluate_model(test_df):
    results = []

    # ThreadPoolExecutor to process rows in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(predictor, row['image_link'], row['group_id'], row['entity_name']) for _, row in test_df.iterrows()]

        # Use tqdm to show progress
        for future in tqdm(futures, total=len(futures), desc="Extracting data", unit="image"):
            results.append(future.result())

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Merge predictions with the test data
    merged_df = test_df.merge(df_results, left_on='group_id', right_on='index')

    # Filter out rows with None predictions
    merged_df = merged_df[merged_df['prediction'].notna()]
    
    # Calculate evaluation metrics
    y_true = merged_df['entity_value']
    y_pred = merged_df['prediction']

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")

    return merged_df

if __name__ == "__main__":
    # Load the dataset from sample_test.csv
    csv_file_path = "./dataset/test.csv"
    df = pd.read_csv(csv_file_path)

    # Extract text and relevant units for each image using parallel processing
    results = []

    # ThreadPoolExecutor to process rows in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(predictor, row['image_link'], row['group_id'], row['entity_name']) for _, row in df.iterrows()]

        # Use tqdm to show progress
        for future in tqdm(futures, total=len(futures), desc="Extracting data", unit="image"):
            results.append(future.result())

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Save the results to a new CSV file
    output_filename = 'test_out.csv'
    df_results.to_csv(output_filename, index=False)
    # evaluate_model(df)
    
    print("Processing complete. Results saved to test_out.csv.")