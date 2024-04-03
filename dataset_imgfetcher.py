import argparse
import os
import requests


def fetch_image_urls(query, num_images, flickr_key):
    url = f"https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key={flickr_key}&text={query}_animal&extras=url_m&per_page={num_images}&format=json&nojsoncallback=1"
    response = requests.get(url)
    data = response.json()

    print(data)

    image_urls = []
    for photo in data['photos']['photo']:
        farm_id = photo['farm']
        server_id = photo['server']
        photo_id = photo['id']
        secret = photo['secret']
        image_url = f"https://farm{farm_id}.staticflickr.com/{server_id}/{photo_id}_{secret}.jpg"
        image_urls.append(image_url)

    return image_urls

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("--data_origin_path", required=True,
        help="Path to data")
    parser.add_argument("--flickr_api_key", required=True,
        help="Flickr API key")
    return parser.parse_args()

def download_images(image_urls, folder_path, label):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, url in enumerate(image_urls):
        try:
            image_path = os.path.join(folder_path, f"image_{label}_{i}.jpg")
            response = requests.get(url)
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded image {i+1}/{len(image_urls)}")
        except Exception as e:
            print(f"Error downloading image {i+1}: {e}")

def get_labels(origin_path, flickr_key):
    _, dir, _ = next(os.walk(origin_path))

    dir_cons = dir

    for label in dir_cons:
        img_urls = fetch_image_urls(label, 140, flickr_key)

        download_images(image_urls=img_urls, folder_path=f"{origin_path}/{label}/", label=label)


args = parse_arguments()

get_labels(args.data_origin_path, args.flickr_api_key)