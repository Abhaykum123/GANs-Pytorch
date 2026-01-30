import glob
import os
import random
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    r"""
    Custom PyTorch Dataset for MNIST-style images
    Folder structure expected:
    train/
        0/
        1/
        2/
        ...
    """
    
    def __init__(self, split, im_path, im_ext='png'):
        """
        Initialize dataset parameters and load image paths.

        Args:
            split (str): Dataset split name (train / test)
            im_path (str): Root directory containing image folders
            im_ext (str): Image file extension (default: png)
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        im.close()
        
        # # Uncomment below 4 lines for colored mnist images
        # a = (im_tensor[0]*random.uniform(0.2, 1.0)).unsqueeze(0)
        # b = (im_tensor[0]*random.uniform(0.2, 1.0)).unsqueeze(0)
        # c = (im_tensor[0]*random.uniform(0.2, 1.0)).unsqueeze(0)
        # im_tensor = torch.cat([a, b, c], dim=0)
        
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
    
if __name__ == "__main__":

    # Root directory containing training images
    image_root_path = r"D:\gan\data\train"

    try:
        # Create dataset instance
        mnist_dataset = MnistDataset(
            split="train",
            im_path=image_root_path,
            im_ext="png"
        )

        # Print dataset size
        print(f"Total images loaded: {len(mnist_dataset)}")

        # Fetch a sample image
        if len(mnist_dataset) > 0:
            sample = mnist_dataset[0]
            print(f"Sample tensor shape: {sample.shape}")  # [1, 28, 28]
            print(f"Tensor value range: {sample.min().item()} to {sample.max().item()}")

    except AssertionError as e:
        print(f"Path error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")