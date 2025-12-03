import torchvision
import os

def download_gtsrb():
    print("Downloading GTSRB Train set...")
    # This will download the dataset to data/gtsrb
    torchvision.datasets.GTSRB(root='data', split='train', download=True)
    print("Downloading GTSRB Test set...")
    torchvision.datasets.GTSRB(root='data', split='test', download=True)
    print("Done!")

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    download_gtsrb()
