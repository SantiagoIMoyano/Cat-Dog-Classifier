import os
import zipfile
import urllib.request

def download_and_unpack(url: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    
    zip_path = os.path.join(extract_to, "cats_and_dogs.zip")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

def get_dirs(base_path: str):
    train_dir = os.path.join(base_path, "train")
    val_dir   = os.path.join(base_path, "validation")
    test_dir  = os.path.join(base_path, "test")
    return train_dir, val_dir, test_dir

def count_images(dir_path: str):
    return sum(len(files) for _, _, files in os.walk(dir_path))