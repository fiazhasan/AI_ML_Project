"""
Data analysis and EDA utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Data analysis and visualization"""
    
    def __init__(self, dataset):
        """
        Initialize analyzer
        
        Args:
            dataset: Dataset instance
        """
        self.dataset = dataset
        self.samples = dataset.samples
    
    def analyze_dataset(self) -> Dict:
        """
        Perform comprehensive dataset analysis
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing dataset...")
        
        # Basic statistics
        total_samples = len(self.samples)
        num_classes = self.dataset.num_classes
        
        # Class distribution
        labels = [s[1] for s in self.samples]
        class_counts = Counter(labels)
        
        # Calculate statistics
        class_sizes = list(class_counts.values())
        mean_samples = np.mean(class_sizes)
        std_samples = np.std(class_sizes)
        min_samples = np.min(class_sizes)
        max_samples = np.max(class_sizes)
        
        # Image statistics (sample a few)
        image_stats = self._analyze_images()
        
        analysis = {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'mean_samples_per_class': mean_samples,
            'std_samples_per_class': std_samples,
            'min_samples_per_class': min_samples,
            'max_samples_per_class': max_samples,
            'class_distribution': dict(class_counts),
            'image_stats': image_stats,
            'imbalance_ratio': max_samples / min_samples if min_samples > 0 else 0
        }
        
        logger.info(f"Analysis complete: {total_samples} samples, {num_classes} classes")
        return analysis
    
    def _analyze_images(self, sample_size: int = 100) -> Dict:
        """
        Analyze image characteristics
        
        Args:
            sample_size: Number of images to sample
            
        Returns:
            Dictionary with image statistics
        """
        import random
        from PIL import Image
        
        sample_indices = random.sample(range(len(self.samples)), min(sample_size, len(self.samples)))
        
        widths = []
        heights = []
        
        for idx in sample_indices:
            img_path, _ = self.samples[idx]
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
            except:
                continue
        
        return {
            'mean_width': np.mean(widths) if widths else 0,
            'mean_height': np.mean(heights) if heights else 0,
            'min_width': np.min(widths) if widths else 0,
            'max_width': np.max(widths) if widths else 0,
            'min_height': np.min(heights) if heights else 0,
            'max_height': np.max(heights) if heights else 0,
        }
    
    def plot_class_distribution(self, save_path: Optional[str] = None):
        """
        Plot class distribution
        
        Args:
            save_path: Path to save plot
        """
        labels = [s[1] for s in self.samples]
        class_counts = Counter(labels)
        
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        class_names = [self.dataset.idx_to_class[c] for c in classes]
        
        plt.figure(figsize=(15, 6))
        plt.bar(range(len(classes)), counts)
        plt.xlabel('Class Index')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        plt.xticks(range(len(classes)), class_names, rotation=90, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_imbalance(self, save_path: Optional[str] = None):
        """
        Plot class imbalance visualization
        
        Args:
            save_path: Path to save plot
        """
        labels = [s[1] for s in self.samples]
        class_counts = Counter(labels)
        
        counts = list(class_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=30, edgecolor='black')
        plt.xlabel('Samples per Class')
        plt.ylabel('Number of Classes')
        plt.title('Class Size Distribution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir: str = "logs"):
        """
        Generate comprehensive analysis report
        
        Args:
            output_dir: Directory to save reports
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Perform analysis
        analysis = self.analyze_dataset()
        
        # Create visualizations
        self.plot_class_distribution(f"{output_dir}/class_distribution.png")
        self.plot_class_imbalance(f"{output_dir}/class_imbalance.png")
        
        # Save text report
        report_path = Path(output_dir) / "data_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Samples: {analysis['total_samples']}\n")
            f.write(f"Number of Classes: {analysis['num_classes']}\n")
            f.write(f"Mean Samples per Class: {analysis['mean_samples_per_class']:.2f}\n")
            f.write(f"Std Samples per Class: {analysis['std_samples_per_class']:.2f}\n")
            f.write(f"Min Samples per Class: {analysis['min_samples_per_class']}\n")
            f.write(f"Max Samples per Class: {analysis['max_samples_per_class']}\n")
            f.write(f"Imbalance Ratio: {analysis['imbalance_ratio']:.2f}\n\n")
            
            f.write("Image Statistics:\n")
            img_stats = analysis['image_stats']
            f.write(f"  Mean Width: {img_stats['mean_width']:.2f}\n")
            f.write(f"  Mean Height: {img_stats['mean_height']:.2f}\n")
            f.write(f"  Width Range: {img_stats['min_width']} - {img_stats['max_width']}\n")
            f.write(f"  Height Range: {img_stats['min_height']} - {img_stats['max_height']}\n")
        
        logger.info(f"Analysis report saved to {output_dir}")
