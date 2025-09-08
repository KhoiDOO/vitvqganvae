import argparse
import os
import sys
import numpy as np
import matplotlib
# Use non-interactive backend for better performance when saving to file
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_point_clouds_grid(point_clouds, titles=None, figsize=(15, 15), point_size=10, elev=30, azim=45, save_path=None):
    """
    Plot multiple point clouds in a grid layout.
    
    Args:
        point_clouds: List of point clouds, each of shape (N, 3)
        titles: List of titles for each subplot
        figsize: Figure size (width, height)
        point_size: Size of points in the scatter plot
        elev: Elevation angle in degrees for 3D view
        azim: Azimuth angle in degrees for 3D view
        save_path: If provided, save the figure to this path
    """
    n = len(point_clouds)
    grid_size = int(np.ceil(np.sqrt(n)))
    
    fig = plt.figure(figsize=figsize)
    
    for i, pc in enumerate(point_clouds, 1):
        ax = fig.add_subplot(grid_size, grid_size, i, projection='3d')
        
        # Convert to numpy if it's a torch tensor
        if isinstance(pc, torch.Tensor):
            pc = pc.permute(1, 0).cpu().numpy()
            
        # Ensure we have the correct shape (N, 3)
        if pc.shape[1] > 3:
            pc = pc[:, :3]  # Take only x, y, z coordinates
            
        # Plot the point cloud
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=point_size, alpha=0.8)
        
        # Set title if provided
        if titles and i-1 < len(titles):
            ax.set_title(titles[i-1], fontsize=8)
            
        # Set equal aspect ratio
        max_extent = np.max([
            np.max(pc[:, 0]) - np.min(pc[:, 0]),
            np.max(pc[:, 1]) - np.min(pc[:, 1]),
            np.max(pc[:, 2]) - np.min(pc[:, 2])
        ]) / 2.0
        
        mid_x = (np.max(pc[:, 0]) + np.min(pc[:, 0])) / 2.0
        mid_y = (np.max(pc[:, 1]) + np.min(pc[:, 1])) / 2.0
        mid_z = (np.max(pc[:, 2]) + np.min(pc[:, 2])) / 2.0
        
        ax.set_xlim(mid_x - max_extent, mid_x + max_extent)
        ax.set_ylim(mid_y - max_extent, mid_y + max_extent)
        ax.set_zlim(mid_z - max_extent, mid_z + max_extent)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Remove axis labels for cleaner look
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
    
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize point clouds from a dataset')
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["mesh500_1024", "mesh500_4096"],
                        help="Dataset to use")
    parser.add_argument("--root", type=str, default=None, 
                        help="Path to dataset root directory")
    parser.add_argument("--num_pc", type=int, default=9, 
                        help="Number of point clouds to visualize (default: 9)")
    parser.add_argument("--split", type=str, default="train", 
                        choices=["train", "test"],
                        help="Which split to use (train/test)")
    parser.add_argument("--point_size", type=float, default=5,
                        help="Size of points in the visualization")
    parser.add_argument("--elev", type=float, default=30,
                        help="Elevation angle for 3D view")
    parser.add_argument("--azim", type=float, default=45,
                        help="Azimuth angle for 3D view")
    
    args = parser.parse_args()

    # Load the dataset
    if args.dataset == "mesh500_1024":
        from vitvqganvae.data.hf.mesh500 import get_mesh500_1024
        train_ds, test_ds = get_mesh500_1024(root=args.root, augment=True)
    elif args.dataset == "mesh500_4096":
        from vitvqganvae.data.hf.mesh500 import get_mesh500_4096
        train_ds, test_ds = get_mesh500_4096(root=args.root, augment=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Select the appropriate split
    dataset = train_ds if args.split == "train" else test_ds
    
    # Determine how many point clouds to visualize
    num_samples = min(args.num_pc, len(dataset)) if args.num_pc > 0 else len(dataset)
    
    print(f"Visualizing {num_samples} point clouds from {args.split} split of {args.dataset}")
    
    # Collect point clouds
    point_clouds = []
    titles = []
    
    for i in tqdm(range(num_samples), desc="Loading point clouds"):
        # Get point cloud (assuming dataset returns (points, label) tuple)
        sample = dataset[i]
        if isinstance(sample, (list, tuple)):
            points = sample[0]  # Get points, ignore label
        else:
            points = sample
            
        point_clouds.append(points)
        titles.append(f"Sample {i+1}")
    
    # Create output directory and filename
    output_dir = os.path.join('.samples', args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'pointclouds.pdf')
    
    # Plot the point clouds and save to file
    plot_point_clouds_grid(
        point_clouds, 
        titles=titles,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
        save_path=output_file
    )