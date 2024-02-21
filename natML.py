def importPackages():
    from PIL import Image
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from sklearn.cluster import KMeans
    from IPython.display import clear_output
    import matplotlib.colors as mcolors
    import os
    import cv2 # import computer vision
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

# defin a function generate a neighborhood average image for each pixel in the image
def average_peripheral_neighborhood(img, n):
    # Prepare the resulting image container
    averaged_img = np.zeros_like(img)
    h, w = img.shape[0], img.shape[1]
    
    for y in range(h):
        for x in range(w):
            # List to gather the neighbors
            neighbors = []
            
            # Check the boundary and get the values of the peripheral neighbors
            for i in [-n, n]:
                if 0 <= y + i < h:
                    neighbors.append(img[y + i, x])
                if 0 <= x + i < w:
                    neighbors.append(img[y, x + i])
            
            # Calculate the average if there are any neighbors
            if neighbors:
                averaged_img[y, x] = np.mean(neighbors, axis=0)
            else:
                averaged_img[y, x] = img[y, x]
                
    return averaged_img

# define a function to generate a "megaimage" with the original image and the neighborhood average images
def makeMegaimage(RGBimg, HSVimg, n=3):
    imagecolumns = ['RGB: FAF', 'RGB: GREEN', 'RGB: IR', 'HSV: Hue', 'HSV: Saturation', 'HSV: Brightness']
    megaimage = np.zeros((RGBimg.shape[0], RGBimg.shape[1], 6+n*6), dtype=np.uint8)
    megaimage[:, :, 0:3] = RGBimg
    megaimage[:, :, 3:6] = HSVimg
    if n == 0:
        return megaimage
    for i in range(1, n+1):
        composite_RGBimg = average_peripheral_neighborhood(RGBimg, i)
        composite_HSVimg = average_peripheral_neighborhood(HSVimg, i)
        imagecolumns.append('RGB: FAF, n_hood={}'.format(i))
        imagecolumns.append('RGB: GREEN, n_hood={}'.format(i))
        imagecolumns.append('RGB: IR, n_hood={}'.format(i))
        imagecolumns.append('HSV: Hue, n_hood={}'.format(i))
        imagecolumns.append('HSV: Saturation, n_hood={}'.format(i))
        imagecolumns.append('HSV: Brightness, n_hood={}'.format(i))
        
        megaimage[:, :, 6*i:6*i+3] = composite_RGBimg
        megaimage[:, :, 6*i+3:6*i+6] = composite_HSVimg
    return megaimage, imagecolumns

# Split into train test dev
def split_train_test(files_path):
    train_ratio = 0.8  # Ratio is ratio of files to train on
    np.random.seed(42)  # set random seed

    # make a random list of indices from files to split into train test dev
    possible_indices = np.arange(len(files_path))
    training_indices = np.random.choice(possible_indices, size=int(len(files_path)*train_ratio), replace=False)
    indicies_remaining = [i for i in possible_indices if i not in training_indices]
    test_indices = np.random.choice(indicies_remaining, size=int(len(indicies_remaining)*0.5), replace=False)
    dev_indices = [i for i in indicies_remaining if i not in test_indices]

    print(f'length of training: {len(training_indices)} ratio = {len(training_indices)/len(possible_indices)*100:.2f}%; test: {len(test_indices)} ratio = {len(test_indices)/len(possible_indices)*100:.2f}%; dev: {len(dev_indices)} ratio = {len(dev_indices)/len(possible_indices)*100:.2f}%')
    print(np.sort(training_indices))
    print(np.sort(test_indices))
    print(np.sort(dev_indices))
    return training_indices, test_indices, dev_indices
    
# Function to generate a dataframe with the image data
def generate_CVS_holders(image_folder, image_folders, grader_folder, grader_folders, NeverranonthisPC = False):
    if not NeverranonthisPC:
        return None
    # The code below need not be run if you have already run it once or if you downloaded the companion csv files from the repository
    n_neighbors = 5

    # using n_neighbors, make a megaimage for each image in each folder
    for foldernumber in range(len(image_folders)):
        current_folder = image_folder + '\\' + image_folders[foldernumber]
        print(f'Working on {image_folders[foldernumber]}; {foldernumber+1}/{len(image_folders)}')
        for eye in ['OD', 'OS']:
            # open current folder to see contents
            current_images = os.listdir(current_folder)
            current_images = [i for i in current_images if 'preprocessed' in i]# throw out files that do not contain 'preprocessed'
            current_images = [i for i in current_images if eye in i] # get the relevant images for OD or OS eye
            if len(current_images) < 3:
                print(f'Not enough images in {current_folder}')
                continue
            # Get the OD grader
            grader_csv = f'{image_folders[foldernumber]} {eye}.csv'
            if grader_csv not in grader_folders: # check if the csv exists
                print(f'csv file {grader_csv} not found')
                continue

            grader_csv = pd.read_csv(grader_folder + '\\' + grader_csv)# open the csv as a dataframe
            grader_csv[['X', 'Y']] = grader_csv['X,Y'].str.split(',', expand=True)# split the X, Y coordinates into two columns
            # work with OD first images
            FAFimg = [i for i in current_images if 'FAF' in i][0]
            greenimg = [i for i in current_images if 'green' in i][0]
            IRimg = [i for i in current_images if 'IR' in i][0]

            # make composite image where FAF is red, green is green, and IR is blue
            FAF = cv2.imread(current_folder + '\\' + FAFimg)
            GREEN = cv2.imread(current_folder + '\\' + greenimg)
            IR = cv2.imread(current_folder + '\\' + IRimg)

            composite = np.zeros((IR.shape[0], IR.shape[1], 3), dtype=np.uint8)
            composite[:, :, 0] = FAF[:, :, 2]
            composite[:, :, 1] = GREEN[:, :, 1]
            composite[:, :, 2] = IR[:, :, 0]
            composite_HSV = cv2.cvtColor(composite, cv2.COLOR_BGR2HSV)
            megaimage, imagecolumns = makeMegaimage(composite, composite_HSV, n=n_neighbors)
            # Make a pandas dataframe
            y_indices, x_indices = np.indices((megaimage.shape[0], megaimage.shape[1]))
            allArray = np.dstack((y_indices, x_indices, megaimage)).reshape((-1, megaimage.shape[2] + 2))
            cols = ['Y', 'X'] + imagecolumns
            df = pd.DataFrame(allArray, columns=cols)
            # Make a name column and add it to the dataframe which should be grader_csv
            df['Name'] =  f'{image_folders[foldernumber]} {eye}'
            # where X and Y matches between df and grader_csv, add the grader_csv column Counter to df            
            for row_idx in range(len(grader_csv)):
                x = int(grader_csv.iloc[row_idx]['X'])
                y = int(grader_csv.iloc[row_idx]['Y'])
                classifier = int(grader_csv.iloc[row_idx]['Counter'])
                df.loc[(df['X'] == x) & (df['Y'] == y), 'Classifier'] = classifier
            # save the dataframe as a csv in folder 'I:\Natalie\Images and grader results\Extracted_and_expanded'
            df.to_csv(f'I:\\Natalie\\Images and grader results\\Extracted_and_expanded\\{image_folders[foldernumber]} {eye}.csv', index=False)

# Function to add columns to the dataframe that are 
def add_columns_to_model(basecols = ['RGB: FAF', 'RGB: GREEN', 'RGB: IR', 'HSV: Hue', 'HSV: Saturation', 'HSV: Brightness'], add_HSV = False, n_neighborhoods = 5):
    
    if add_HSV:
        cols_to_add = basecols
    else: # if add_HSV is False, then only add the RGB columns
        cols_to_add = [i for i in basecols if 'RGB' in i]
    
    n_basecols = len(cols_to_add)
    
    if n_neighborhoods >5:
        ValueError('n_neighborhoods must be less than or equal to 5')
    else:
        for i in range(1, n_neighborhoods+1):
            cols_to_add.append(f'RGB: FAF, n_hood={i}')
            cols_to_add.append(f'RGB: GREEN, n_hood={i}')
            cols_to_add.append(f'RGB: IR, n_hood={i}')
            if add_HSV:
                cols_to_add.append(f'HSV: Hue, n_hood={i}')
                cols_to_add.append(f'HSV: Saturation, n_hood={i}')
                cols_to_add.append(f'HSV: Brightness, n_hood={i}')
        return cols_to_add, n_basecols

# Because our data has n_neighbors and is built up of 3 columns for RGB and another 3 for HSV, we can choose what columns to use for the training.
# You can open any csv in the \Extracted_and_expanded folder to see the column names.
def init_trainingArray(cols_to_add, training_idx, extracted_path, files):
    training_nparray = np.zeros((0, len(cols_to_add)))
    print(f'Columns: {cols_to_add}')
    for i in range(len(training_idx)):
        temp_df = pd.read_csv(extracted_path + '\\' + files[training_idx[i]])
        temp_df = temp_df[cols_to_add].to_numpy()
        training_nparray = np.vstack((training_nparray, temp_df))
        print(f'Array shape: {training_nparray.shape}, {i+1}/{len(training_idx)}')

    # Below are flags that tell us that initialization has changed and we have not scaled or weighted the data yet.
    scaling_performed_Flag = False
    weightingperformed_Flag = False

    return training_nparray, scaling_performed_Flag, weightingperformed_Flag


def scaleArray(training_nparray, ScalingChoice = 1):
    
    if ScalingChoice in [1, 2]:
        scaling_performed_Flag = True
        if ScalingChoice == 1:#perform standardization
            scaler = StandardScaler()
            scalingmethod_performed = 'StandardScaler'

        if ScalingChoice == 2:#perfomr scaling minmax
            scaler = MinMaxScaler()
            # each column is scaled independently
            scalingmethod_performed = 'MinMaxScaler'
        training_nparray_scaled = scaler.fit_transform(training_nparray)# each column is standardized independently
        print(f'Scaling: {scalingmethod_performed}')
    else:
        ValueError('ScalingChoice must be 1 for StandardScaler or 2 for MinMaxScaler')
    return training_nparray_scaled, scaling_performed_Flag
   
    
def create_weighing_array(n_basecols, cols_to_add, weight_baseline = None, gamma = 0.8):
    if weight_baseline is None:
        weight_baseline = [1]*n_basecols
        
    weights = [i/max(weight_baseline) for i in weight_baseline]
    # normalize the weights
    
    n_range = len(weight_baseline)
    
    for i in range(int((len(cols_to_add)-n_basecols)/n_basecols)):  # NEEDS FIXING
        for n in range(n_range):
        ########## NEEDS FIXING
            weights.append(weight_baseline[n] * gamma**(i+1)) # NEEDS FIXING
        ############################################
    return weights


def weightArray(training_nparray, weights):
    weighted_training_nparray = np.zeros_like(training_nparray)
    for i in range(training_nparray.shape[1]):
        weighted_training_nparray[:,i] = training_nparray[:,i] * weights[i]
    return weighted_training_nparray
    
    
def plot_alteredarray(original_array, altered_array):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(original_array[1:5000,:], ax=ax1)
    ax1.set_title('Original array')
    sns.heatmap(altered_array[1:5000,:], ax=ax2)
    ax2.set_title('Altered array')



