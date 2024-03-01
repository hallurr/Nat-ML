# Nat-ML: Image Analysis and Clustering Toolkit

## Overview
This project provides a suite of tools for image analysis, focusing on clustering techniques to analyze and interpret distributions within images. It leverages machine learning, statistical methods, and image processing libraries to extract meaningful insights from image data. The toolkit includes functionality for preprocessing images, generating features through neighborhood averaging, creating composite images, and applying clustering algorithms to segment images based on various features.

## Features
- **Image Preprocessing**: Utilities to preprocess images for analysis, including color space transformations and peripheral neighborhood averaging.
- **Feature Generation**: Generate a "megaimage" that combines original and neighborhood-averaged images to enrich feature space for clustering.
- **Clustering**: Apply KMeans clustering to segment images based on the generated features, supported by metrics to evaluate clustering quality.
- **Statistical Analysis**: Functions to calculate likelihoods using Euclidean distances, cosine similarity, and KL divergence among distributions.
- **Visualization**: Tools for visualizing the original images, the effects of preprocessing, and the results of clustering.

## Dependencies
- Python 3.x
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [OpenCV](https://opencv.org/)
- [SciPy](https://www.scipy.org/)
- [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)

## Installation
To use this toolkit, ensure you have Python 3.x installed on your system. Then, install the required dependencies using pip:

## Usage
Below is a brief overview of how to use the core functionality in this toolkit. For detailed examples and documentation, refer to the [Wiki](https://github.com/your-repository/wiki).

## Contributing
We welcome contributions to this project! Please contact me for details on how to submit contributions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

# Using the IPYNB

## Introduction
This notebook guides you through the process of analyzing images using clustering techniques. It demonstrates how to pick metrics, assess their effectiveness, and visualize the results. The goal is to provide a comprehensive toolkit for image feature extraction, clustering, and evaluation.

## Setup
First, ensure you have all necessary libraries installed and the natML.py file in your working directory. The natML.py file contains the core functionality used throughout this notebook.

### Import Libraries
run 
```
from natML import *  # Importing custom functions from natML.py
```
#### Define Data Path
Specify the path to your image data. This path should point to a directory containing your image folders and the "Extracted_and_expanded" folder.
```
# Define the directory containing image data and the 'Extracted_and_expanded' folder
dir = r'your\path\here'  # Update this path to your specific directory
```

#### Data Preparation
Before diving into clustering, ensure your data is prepared and organized in the expected structure.

#### Generate CVS Holders
If running for the first time or on a new machine, set NeverranonthisPC to True to generate CVS holders, otherwise keep False.
```
NeverranonthisPC = True
```

## Clustering Analysis
This section demonstrates how to perform clustering on your image data, evaluate different metrics, and select the best parameters for your analysis.

### Set Parameters
Define the parameters for your clustering analysis. Adjust these based on your dataset and analysis needs.

```
# Example Gratis
n_neighborhoods_wanted = 5
HSV_wanted = False
weights_startingpoint = [1, 1, 1.15] # Emphasize the IR channel by 15%
gamma_wanted = 0.7
ScalingChoice_wanted = 1
n_Classifiers = 8

# Changing run for random seed and n_clusters
random_seed_list = [42]
n_clusters_wanted_list = [15]
```

### Run Clustering Analysis
Execute the clustering analysis over a range of seeds and cluster numbers to find the optimal configuration.

### Visualize Results
After running the analysis, visualize the clustering results to assess the effectiveness of your chosen metrics.
```
# Example visualization code
draw_hexplot(RGB_image=..., cluster_img=..., n_clusters=...)
```

#### Interactive Visualization
Leverage IPython widgets to interactively explore the clustering results and adjust parameters in real-time.

```
interactive_plot = interactive(plot_contours, ...)
```

## Conclusion
Summarize the findings from your clustering analysis and provide recommendations for further analysis or application of these techniques.


Have fun!