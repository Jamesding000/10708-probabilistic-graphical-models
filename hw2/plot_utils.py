import numpy as np
import os
import shutil
import tempfile

from matplotlib import pyplot as plt
from PIL import Image

## Some helper functions for visualization and submission
def visualize_state(state, title=None, file_path=None):
    """
    Visualize the 2D state as an image.
    
    Parameters:
        state (np.ndarray): the 2D state to be visualized
        title (str, optional): the title of the plot
        file_path (str, optional): the path to save the image
    """
    assert isinstance(state, np.ndarray)

    N, M = state.shape
    X, Y = np.meshgrid(range(N + 1), range(M + 1))
    ax = plt.axes()
    ax.imshow(state, "binary", vmin=-1, vmax=1)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)      
    
    if title is not None:
        plt.title(title)
    plt.axis('tight')
    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.clf()


def generate_gif(samples, output_path, image_dir=None):
    """
    Generate a GIF using collected samples (not required)
    
    Parameters:
        samples (List of np.ndarray): the list of samples
        output_path: the path to save the GIF
        image_dir (optional): the directory for saving the images,
            if not provided, the images will be removed afterwards
    """
    if image_dir is None:
        image_dir = tempfile.mkdtemp()
        rm_flag = True
    else:
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        rm_flag = False
        
    for i, sample in enumerate(samples):
        visualize_state(sample, f"Time={i}", f"{image_dir}/{i}.png")
    
    images = []
    for i in range(len(samples)):
        images.append(Image.open(f"{image_dir}/{i}.png"))
        
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=20)
    
    if rm_flag:
        shutil.rmtree(image_dir)


def merge_images(filenames, n_rows, n_cols, output_path):
    """
    Merge a list of image files into single image (for submission)
    
    Parameters:
        filenames (list of str): the images which are going to be merged
        n_rows (int): the number of rows in the image grid
        n_cols (int): the number of columns in the image grid
        output_path (str): the output path of the merged image
    """
    images = [Image.open(filename) for filename in filenames]
    width, height = images[0].size
    new_image = Image.new('RGB', (width * n_cols, height * n_rows))
    for row in range(n_rows):
        for col in range(n_cols):
            new_image.paste(
                images[row * n_cols + col],
                (width * col, height * row))
            
    new_image.save(output_path)