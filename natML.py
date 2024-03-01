
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
import time
import pickle
from scipy.spatial import distance
from scipy.stats import entropy
from ipywidgets import interactive
import ipywidgets as widgets

def get_likelihoods(new_array, distributions, epsilon = 1e-10): # new_array = the new array to compare to the cluster distributions
    new_array_no_zeros = np.clip(new_array, epsilon, None)
    new_array_no_zeros /= np.sum(new_array_no_zeros)
    
    distributions_no_zeros = np.clip(distributions, epsilon, None)
    
    # Calculate Euclidean distances
    euclidean_distances = np.linalg.norm(distributions_no_zeros - new_array_no_zeros, axis=1)
    
    # Calculate Cosine Similarity
    cosine_similarity = np.dot(distributions_no_zeros, new_array_no_zeros) / (np.linalg.norm(distributions_no_zeros, axis=1) * np.linalg.norm(new_array_no_zeros))
    
    # Calculate KL Divergence
    kl_divergence = np.array([entropy(new_array_no_zeros, dist_no_zeros) for dist_no_zeros in distributions_no_zeros])
    
    # Normalize Euclidean distances to likelihood
    euclidean_likelihood = 1 / (euclidean_distances + epsilon)
    euclidean_likelihood /= np.sum(euclidean_likelihood)
    
    # Normalize Cosine Similarity to likelihood
    cosine_likelihood = cosine_similarity / np.sum(cosine_similarity)
    
    # Normalize KL Divergence to likelihood
    KL_likelihood = 1 / (kl_divergence + epsilon)
    KL_likelihood /= np.sum(KL_likelihood)
    
    return euclidean_likelihood, cosine_likelihood, KL_likelihood

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
def split_train_test(files_path, random_seed_wanted = 42, train_ratio = 0.8, test_to_dev_ratio = 0.5):
    np.random.seed(random_seed_wanted)  # set random seed

    # make a random list of indices from files to split into train test dev
    possible_indices = np.arange(len(files_path))
    training_indices = np.random.choice(possible_indices, size=int(len(files_path)*train_ratio), replace=False)
    indicies_remaining = [i for i in possible_indices if i not in training_indices]
    test_indices = np.random.choice(indicies_remaining, size=int(len(indicies_remaining)*test_to_dev_ratio), replace=False)
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
def init_trainingArray(cols_to_add, training_idx, extracted_path, files, n_Classifiers = 8, annotate = False):
    training_nparray = np.zeros((0, len(cols_to_add)))
    training_nparray_labels = {}
    for i in range(8):
        training_nparray_labels[i] = np.zeros((0, len(cols_to_add)))
    print(f'Columns: {cols_to_add}')
    for i in range(len(training_idx)):
        temp_df = pd.read_csv(extracted_path + '\\' + files[training_idx[i]])
        # Only take rows where Classifier is not NaN
        for i in range(n_Classifiers):
            temp_df_labelrows = temp_df[temp_df['Classifier'] == i]
            temp_df_labelrows = temp_df_labelrows[cols_to_add].to_numpy()
            training_nparray_labels[i] = np.vstack((training_nparray_labels[i], temp_df_labelrows))
        temp_df = temp_df[cols_to_add].to_numpy()
        training_nparray = np.vstack((training_nparray, temp_df))
        if annotate:
            print(f'Array shape: {training_nparray.shape}, {i+1}/{len(training_idx)}')

    # Below are flags that tell us that initialization has changed and we have not scaled or weighted the data yet.
    if not annotate:
        print(f'Array shape: {training_nparray.shape}')
    return training_nparray, training_nparray_labels


def scaleArray(training_nparray, ScalingChoice = 1):
    
    if ScalingChoice in [1, 2]:
        if ScalingChoice == 1:#perform standardization
            scaler = StandardScaler()
            scalingmethod_performed = 'StandardScaler'

        if ScalingChoice == 2:#perfomr scaling minmax
            scaler = MinMaxScaler()
            # each column is scaled independently
            scalingmethod_performed = 'MinMaxScaler'
        training_nparray_scaled = scaler.fit_transform(training_nparray)# each column is standardized independently
    else:
        ValueError('ScalingChoice must be 1 for StandardScaler or 2 for MinMaxScaler')
    return training_nparray_scaled, scaler
   
    
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


def get_Healthy_vs_Disease_npArrays(training_nparray_labels, test_dev_nparray_labels, n_Classifiers, Healthy_labels = [5, 6], Disease_labels = [0, 3, 4]):
    # Assuming definitions of your variables and n_Classifiers somewhere above this code
    Healthy_training_nparray_labels = np.array([])
    Healthy_test_dev_nparray_labels = np.array([])
    Disease_training_nparray_labels = np.array([])
    Disease_test_dev_nparray_labels = np.array([])

    for i in range(n_Classifiers):
        if i in Healthy_labels:
            # Use .size to check if the array is empty
            if Healthy_training_nparray_labels.size == 0:
                Healthy_training_nparray_labels = training_nparray_labels[f'{i}_clusters']
                Healthy_test_dev_nparray_labels = test_dev_nparray_labels[f'{i}_clusters']
            else:
                Healthy_training_nparray_labels = np.hstack((Healthy_training_nparray_labels, training_nparray_labels[f'{i}_clusters']))
                Healthy_test_dev_nparray_labels = np.hstack((Healthy_test_dev_nparray_labels, test_dev_nparray_labels[f'{i}_clusters']))
        elif i in Disease_labels:
            if Disease_training_nparray_labels.size == 0:
                Disease_training_nparray_labels = training_nparray_labels[f'{i}_clusters']
                Disease_test_dev_nparray_labels = test_dev_nparray_labels[f'{i}_clusters']
            else:
                Disease_training_nparray_labels = np.hstack((Disease_training_nparray_labels, training_nparray_labels[f'{i}_clusters']))
                Disease_test_dev_nparray_labels = np.hstack((Disease_test_dev_nparray_labels, test_dev_nparray_labels[f'{i}_clusters']))
    return Healthy_training_nparray_labels, Healthy_test_dev_nparray_labels, Disease_training_nparray_labels, Disease_test_dev_nparray_labels


# Assess the divergence between the Healthy and Disease clusters for the training
def KL_divergence(A, B, n_clusters_wanted, smoothing_value = 1e-10):
    # A and B are two arrays
    A_probs = np.zeros(n_clusters_wanted)
    B_probs = np.zeros(n_clusters_wanted)
    for i in range(n_clusters_wanted):
        A_probs[i] = np.sum(A == i)/len(A)
        B_probs[i] = np.sum(B == i)/len(B)
        
    # Adding a small constant to avoid division by zero or log(0)
    A_probs += smoothing_value
    B_probs += smoothing_value
    
    # Ensure normalization after adding smoothing_value
    A_probs /= A_probs.sum()
    B_probs /= B_probs.sum()
        
    KL_divergence = np.sum(A_probs * np.log(A_probs/B_probs))
    return KL_divergence


def Mega_Run_seedPortion(extracted_path,
                     files,
                     n_Classifiers,
                     n_neighborhoods_wanted,
                     HSV_wanted,
                     weights_startingpoint,
                     gamma_wanted,
                     random_seed,
                     ScalingChoice_wanted,
                     annotate = False,):
    MegaDict = {} # initialize the dictionary
    timer_0 = time.time() # start the timer
    # Split into train, test, dev
    if annotate:
        print(f'Splitting into train, test, dev; Timepassed = {time.time()-timer_0:.2f}s')
    training_indices, test_indices, dev_indices = split_train_test(files_path = files, 
                                                                random_seed_wanted = random_seed, 
                                                                train_ratio = 0.8, 
                                                                test_to_dev_ratio = 0.5)

    # Add columns to the model (using 5 neighborhoods)
    cols_to_add, n_basecols = add_columns_to_model(n_neighborhoods = n_neighborhoods_wanted, 
                                                add_HSV = HSV_wanted)

    # Initialize the trainingarray

    if annotate:
        print(f'Extracting and adding columns to the training array; Timepassed = {time.time()-timer_0:.2f}s')
    training_nparray, training_nparray_labels = init_trainingArray(cols_to_add = cols_to_add, 
                                        training_idx = training_indices, 
                                        extracted_path = f'{dir}\\Extracted_and_expanded', 
                                        files = os.listdir(extracted_path), 
                                        n_Classifiers = n_Classifiers)

    # Add together the Test and dev indices
    
    if annotate:
        print(f'Extracting and adding columns to the test_dev array; Timepassed = {time.time()-timer_0:.2f}s')
    test_dev_indices = np.concatenate((test_indices, dev_indices))
    test_dev_nparray, test_dev_nparray_labels = init_trainingArray(cols_to_add = cols_to_add, 
                                        training_idx = test_dev_indices, 
                                        extracted_path = f'{dir}\\Extracted_and_expanded', 
                                        files = os.listdir(extracted_path),
                                        n_Classifiers = n_Classifiers)

    # Scale the trainingarray
    
    if annotate:
        print(f'Scaling the training array; Timepassed = {time.time()-timer_0:.2f}s')
    training_nparray_scaled, scaler_fitted = scaleArray(training_nparray = training_nparray, 
                                                                ScalingChoice = ScalingChoice_wanted)
    # Scale the test_dev array
    
    if annotate:
        print(f'Scaling the test_dev array; Timepassed = {time.time()-timer_0:.2f}s')
    test_dev_nparray_scaled = scaler_fitted.transform(test_dev_nparray)

    
    if annotate:
        print(f'Scaling the labels; Timepassed = {time.time()-timer_0:.2f}s')
    for i in range(n_Classifiers):
        training_nparray_labels[i] = scaler_fitted.transform(training_nparray_labels[i])
        test_dev_nparray_labels[i] = scaler_fitted.transform(test_dev_nparray_labels[i])

    # Generate weights for the trainingarray
    
    if annotate:
        print(f'Generating weights for the training array; Timepassed = {time.time()-timer_0:.2f}s')
    weights = create_weighing_array(n_basecols = n_basecols, 
                                    cols_to_add = cols_to_add, 
                                    weight_baseline = weights_startingpoint, 
                                    gamma = gamma_wanted) # Decay by 20% for each increase in neighborhood by 1

    # perform the weighting of the trainingarray
    if annotate:
        print(f'Weighting the training array; Timepassed = {time.time()-timer_0:.2f}s')
    training_nparray_scaled_weighted = weightArray(training_nparray = training_nparray_scaled,
                                                weights = weights)
    # perform the weighting of the test_dev array
    if annotate:
        print(f'Weighting the test_dev array; Timepassed = {time.time()-timer_0:.2f}s')
    test_dev_nparray_scaled_weighted = weightArray(training_nparray = test_dev_nparray_scaled,
                                                weights = weights)
    if annotate:
        print(f'Plotting the original and altered array; Timepassed = {time.time()-timer_0:.2f}s')
    for i in range(n_Classifiers):
        training_nparray_labels[i] = weightArray(training_nparray = training_nparray_labels[i],
                                                weights = weights)
        test_dev_nparray_labels[i] = weightArray(training_nparray = test_dev_nparray_labels[i],
                                                weights = weights)

    # Plot the original and altered array
    if annotate:
        print(f'Plotting the original and altered array; Timepassed = {time.time()-timer_0:.2f}s')
        plot_alteredarray(original_array = training_nparray_scaled, 
                        altered_array = training_nparray_scaled_weighted)
        
    return training_nparray_scaled_weighted, training_nparray_labels, test_dev_nparray_scaled_weighted, test_dev_nparray_labels, scaler_fitted, cols_to_add


def Mega_Run_Cluster_Portion(training_nparray_scaled_weighted,
                             training_nparray_labels,
                             
                             test_dev_nparray_scaled_weighted,
                             test_dev_nparray_labels,
                             n_Classifiers,
                             n_clusters_wanted,
                             n_neighborhoods_wanted,
                             
                             weights_startingpoint,
                             gamma_wanted,
                             ScalingChoice_wanted,
                             random_seed,
                             scaler_fitted, 
                             cols_to_add,
                             annotate = False,
                             wantSave = False,
                             colormap = 'bone'):
    
    # train the kmeans model on the training_nparray with the wanted metrics (n_clusters_wanted = 6) and random_state = 0 as defaults
    if annotate:
        print(f'Training the KMeans model: n_clusters = {n_clusters_wanted}, random_state = {random_seed}')
    KMeansclus = KMeans(n_clusters=n_clusters_wanted, random_state=random_seed).fit(training_nparray_scaled_weighted)
    # Predict the clusters for the test_dev array
    MegaDict = {}
    if annotate:
        print(f'Predicting the clusters for the training array')
    train_clusters = KMeansclus.predict(training_nparray_scaled_weighted)
    MegaDict['train_clusters'] = train_clusters
    if annotate:
        print(f'Predicting the clusters for the test_dev array')
    test_dev_clusters = KMeansclus.predict(test_dev_nparray_scaled_weighted)
    MegaDict['test_dev_clusters'] = test_dev_clusters

    
    for i in range(n_Classifiers):
        training_nparray_labels[f'{i}_clusters'] = KMeansclus.predict(training_nparray_labels[i])
        test_dev_nparray_labels[f'{i}_clusters'] = KMeansclus.predict(test_dev_nparray_labels[i])

    MegaDict['training_nparray_labels'] = training_nparray_labels
    MegaDict['test_dev_nparray_labels'] = test_dev_nparray_labels
    
    # Assess KL divergence between ALL pixels in the training, and ALL pixels in the test_dev
    KL_divergence_Train_vs_test_dev = KL_divergence(train_clusters, test_dev_clusters, n_clusters_wanted)

    # Assess the KL divergence for every label
    # First within the training
    KL_divergence_training_labels = np.zeros((n_Classifiers, n_Classifiers))
    # Then within the test_dev
    KL_divergence_test_dev_labels = np.zeros((n_Classifiers, n_Classifiers))
    # Then between the training and test_dev
    KL_divergence_training_test_dev_labels = np.zeros((n_Classifiers, n_Classifiers))
    for i in range(n_Classifiers):
        for j in range(n_Classifiers):
            if i == j:
                KL_divergence_training_labels[i,j] = np.nan
                KL_divergence_test_dev_labels[i,j] = np.nan
                KL_divergence_training_test_dev_labels[i,j] = np.nan
            KL_divergence_training_labels[i,j] = KL_divergence(training_nparray_labels[f'{i}_clusters'], training_nparray_labels[f'{j}_clusters'], n_clusters_wanted)
            KL_divergence_test_dev_labels[i,j] = KL_divergence(test_dev_nparray_labels[f'{i}_clusters'], test_dev_nparray_labels[f'{j}_clusters'], n_clusters_wanted)
            KL_divergence_training_test_dev_labels[i,j] = KL_divergence(training_nparray_labels[f'{i}_clusters'], test_dev_nparray_labels[f'{j}_clusters'], n_clusters_wanted)

    # store this data in a dictionary, keys are labelled with n_clusters_wanted, random_seed, n_neighborhoods_wanted, HSV_wanted, weights_startingpoint, gamma_wanted, ScalingChoice_wanted
    key_string = f'n_clusters = {n_clusters_wanted}, random_seed = {random_seed}, n_neighborhoods = {n_neighborhoods_wanted}, HSV = {HSV_wanted}, weights = {weights_startingpoint}, gamma = {gamma_wanted}, ScalingChoice = {ScalingChoice_wanted}'
    
    MegaDict[f'KL_divergence_Train_vs_test_dev_{key_string}'] = KL_divergence_Train_vs_test_dev
    MegaDict[f'KL_divergence_training_labels_{key_string}'] = KL_divergence_training_labels
    MegaDict[f'KL_divergence_test_dev_labels_{key_string}'] = KL_divergence_test_dev_labels
    MegaDict[f'KL_divergence_training_test_dev_labels_{key_string}'] = KL_divergence_training_test_dev_labels

        
    Healthy_training_l, Healthy_test_dev_l, Disease_training_l, Disease_test_dev_l = get_Healthy_vs_Disease_npArrays(training_nparray_labels = training_nparray_labels, 
                                                                                                                    test_dev_nparray_labels = test_dev_nparray_labels, 
                                                                                                                    n_Classifiers = n_Classifiers, 
                                                                                                                    Healthy_labels = [5, 6], 
                                                                                                                    Disease_labels = [0, 3, 4])

    MegaDict['Healthy_training_l'] = Healthy_training_l
    MegaDict['Healthy_test_dev_l'] = Healthy_test_dev_l
    MegaDict['Disease_training_l'] = Disease_training_l
    MegaDict['Disease_test_dev_l'] = Disease_test_dev_l
    
    
    KL_divergence_train_HealthvsDisease = KL_divergence(Healthy_training_l, Disease_training_l, n_clusters_wanted)
    KL_divergence_test_dev_HealthvsDisease = KL_divergence(Healthy_test_dev_l, Disease_test_dev_l, n_clusters_wanted)
    KL_divergence_healthy_trainvstestdev = KL_divergence(Healthy_training_l, Healthy_test_dev_l, n_clusters_wanted)
    KL_divergence_disease_trainvstestdev = KL_divergence(Disease_training_l, Disease_test_dev_l, n_clusters_wanted)

    MegaDict[f'KL_divergence_train_HealthvsDisease_{key_string}'] = KL_divergence_train_HealthvsDisease
    MegaDict[f'KL_divergence_test_dev_HealthvsDisease_{key_string}'] = KL_divergence_test_dev_HealthvsDisease
    MegaDict[f'KL_divergence_healthy_trainvstestdev_{key_string}'] = KL_divergence_healthy_trainvstestdev
    MegaDict[f'KL_divergence_disease_trainvstestdev_{key_string}'] = KL_divergence_disease_trainvstestdev

    KL_singles = [KL_divergence_Train_vs_test_dev,\
                KL_divergence_train_HealthvsDisease,\
                    KL_divergence_test_dev_HealthvsDisease,\
                        KL_divergence_healthy_trainvstestdev,\
                            KL_divergence_disease_trainvstestdev]

    # save the dict as a pickle with the name f'Megadict_nClus_{n_clusters_wanted}_seed_{random_seed}_nhoods_{n_neighborhoods_wanted}.pkl'
    
    savename = f'Megadict_nClus_{n_clusters_wanted}_seed_{random_seed}_nhoods_{n_neighborhoods_wanted}.pkl'
    if wantSave:
        savename = f'Megadict_nClus_{n_clusters_wanted}_seed_{random_seed}_nhoods_{n_neighborhoods_wanted}.pkl'
        with open(savename, 'wb') as f:
            pickle.dump(MegaDict, f)


    # Create a figure for the KL divergences
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    sns.heatmap(KL_divergence_training_labels, ax=ax[0, 0], cmap=colormap, annot=True, fmt='.2f')
    # Average value of the KL divergence
    aveKL1 = np.nanmean(KL_divergence_training_labels)
    stdKL1 = np.nanstd(KL_divergence_training_labels)
    ax[0, 0].set_title(f'Training: Internal Label Differences\nMean = {aveKL1:.2f} ± {stdKL1:.2f}')
    sns.heatmap(KL_divergence_test_dev_labels, ax=ax[0, 1], cmap=colormap, annot=True, fmt='.2f')
    aveKL2 = np.nanmean(KL_divergence_test_dev_labels)
    stdKL2 = np.nanstd(KL_divergence_test_dev_labels)
    ax[0, 1].set_title(f'Test_dev: Internal Label Differences\nMean = {aveKL2:.2f} ± {stdKL2:.2f}')
    sns.heatmap(KL_divergence_training_test_dev_labels, ax=ax[1, 0], cmap=colormap, annot=True, fmt='.2f')
    aveKL3 = np.nanmean(KL_divergence_training_test_dev_labels)
    stdKL3 = np.nanstd(KL_divergence_training_test_dev_labels)
    ax[1, 0].set_title(f'Training vs Test_dev: Label Differences\nMean = {aveKL3:.2f} ± {stdKL3:.2f}')
    ax[1, 1].plot(range(5), KL_singles, marker='o')
    # Label each datapoint with the value
    for i in range(5):
        ax[1, 1].text(i, KL_singles[i], f'{KL_singles[i]:.2f}', ha='center', va='bottom')
    # set the xticks to be the names of the KL_singles
    ax[1, 1].set_xticks(range(5))
    ax[1, 1].set_xticklabels(['Train vs Test_dev', 'Train_Health vs Disease', 'Test_dev_Health vs Disease', 'Healthy Train vs Test_dev', 'Disease Train vs Test_dev'])
    ax[1, 1].tick_params(axis='x', rotation=45)
    ax[1, 1].set_title('Single KL divergences')
    plt.suptitle(f'{savename[:-4]}')
    plt.tight_layout()

    # Save the image
    fig.savefig(f'{savename[:-4]}.png')

    time.sleep(1)
    
    plt.close('all')  
    
    return MegaDict, KMeansclus, scaler_fitted, cols_to_add, key_string


def Mega_Run(extracted_path,
                     files,
                     n_Classifiers,
                     n_neighborhoods_wanted,
                     HSV_wanted,
                     weights_startingpoint,
                     gamma_wanted,
                     random_seed,
                     n_clusters_wanted,
                     ScalingChoice_wanted,
                     annotate = False,
                     wantSave = False,
                     colormap = 'bone'):
    MegaDict = {} # initialize the dictionary
    timer_0 = time.time() # start the timer
    # Split into train, test, dev
    if annotate:
        print(f'Splitting into train, test, dev; Timepassed = {time.time()-timer_0:.2f}s')
    training_indices, test_indices, dev_indices = split_train_test(files_path = files, 
                                                                random_seed_wanted = random_seed, 
                                                                train_ratio = 0.8, 
                                                                test_to_dev_ratio = 0.5)

    # Add columns to the model (using 5 neighborhoods)
    cols_to_add, n_basecols = add_columns_to_model(n_neighborhoods = n_neighborhoods_wanted, 
                                                add_HSV = HSV_wanted)

    # Initialize the trainingarray

    if annotate:
        print(f'Extracting and adding columns to the training array; Timepassed = {time.time()-timer_0:.2f}s')
    training_nparray, training_nparray_labels = init_trainingArray(cols_to_add = cols_to_add, 
                                        training_idx = training_indices, 
                                        extracted_path = f'{dir}\\Extracted_and_expanded', 
                                        files = os.listdir(extracted_path), 
                                        n_Classifiers = n_Classifiers)

    # Add together the Test and dev indices
    
    if annotate:
        print(f'Extracting and adding columns to the test_dev array; Timepassed = {time.time()-timer_0:.2f}s')
    test_dev_indices = np.concatenate((test_indices, dev_indices))
    test_dev_nparray, test_dev_nparray_labels = init_trainingArray(cols_to_add = cols_to_add, 
                                        training_idx = test_dev_indices, 
                                        extracted_path = f'{dir}\\Extracted_and_expanded', 
                                        files = os.listdir(extracted_path),
                                        n_Classifiers = n_Classifiers)

    # Scale the trainingarray
    
    if annotate:
        print(f'Scaling the training array; Timepassed = {time.time()-timer_0:.2f}s')
    training_nparray_scaled, scaler_fitted = scaleArray(training_nparray = training_nparray, 
                                                                ScalingChoice = ScalingChoice_wanted)
    # Scale the test_dev array
    
    if annotate:
        print(f'Scaling the test_dev array; Timepassed = {time.time()-timer_0:.2f}s')
    test_dev_nparray_scaled = scaler_fitted.transform(test_dev_nparray)

    
    if annotate:
        print(f'Scaling the labels; Timepassed = {time.time()-timer_0:.2f}s')
    for i in range(n_Classifiers):
        training_nparray_labels[i] = scaler_fitted.transform(training_nparray_labels[i])
        test_dev_nparray_labels[i] = scaler_fitted.transform(test_dev_nparray_labels[i])

    # Generate weights for the trainingarray
    
    if annotate:
        print(f'Generating weights for the training array; Timepassed = {time.time()-timer_0:.2f}s')
    weights = create_weighing_array(n_basecols = n_basecols, 
                                    cols_to_add = cols_to_add, 
                                    weight_baseline = weights_startingpoint, 
                                    gamma = gamma_wanted) # Decay by 20% for each increase in neighborhood by 1

    # perform the weighting of the trainingarray
    if annotate:
        print(f'Weighting the training array; Timepassed = {time.time()-timer_0:.2f}s')
    training_nparray_scaled_weighted = weightArray(training_nparray = training_nparray_scaled,
                                                weights = weights)
    # perform the weighting of the test_dev array
    if annotate:
        print(f'Weighting the test_dev array; Timepassed = {time.time()-timer_0:.2f}s')
    test_dev_nparray_scaled_weighted = weightArray(training_nparray = test_dev_nparray_scaled,
                                                weights = weights)
    if annotate:
        print(f'Plotting the original and altered array; Timepassed = {time.time()-timer_0:.2f}s')
    for i in range(n_Classifiers):
        training_nparray_labels[i] = weightArray(training_nparray = training_nparray_labels[i],
                                                weights = weights)
        test_dev_nparray_labels[i] = weightArray(training_nparray = test_dev_nparray_labels[i],
                                                weights = weights)

    # Plot the original and altered array
    if annotate:
        print(f'Plotting the original and altered array; Timepassed = {time.time()-timer_0:.2f}s')
        plot_alteredarray(original_array = training_nparray_scaled, 
                        altered_array = training_nparray_scaled_weighted)


    # train the kmeans model on the training_nparray with the wanted metrics (n_clusters_wanted = 6) and random_state = 0 as defaults
    if annotate:
        print(f'Training the KMeans model: n_clusters = {n_clusters_wanted}, random_state = {random_seed}')
    KMeansclus = KMeans(n_clusters=n_clusters_wanted, random_state=random_seed).fit(training_nparray_scaled_weighted)
    # Predict the clusters for the test_dev array
    if annotate:
        print(f'Predicting the clusters for the training array')
    train_clusters = KMeansclus.predict(training_nparray_scaled_weighted)
    if annotate:
        print(f'Predicting the clusters for the test_dev array')
    test_dev_clusters = KMeansclus.predict(test_dev_nparray_scaled_weighted)

    for i in range(n_Classifiers):
        training_nparray_labels[f'{i}_clusters'] = KMeansclus.predict(training_nparray_labels[i])
        test_dev_nparray_labels[f'{i}_clusters'] = KMeansclus.predict(test_dev_nparray_labels[i])

    # Assess KL divergence between ALL pixels in the training, and ALL pixels in the test_dev
    KL_divergence_Train_vs_test_dev = KL_divergence(train_clusters, test_dev_clusters, n_clusters_wanted)

    # Assess the KL divergence for every label
    # First within the training
    KL_divergence_training_labels = np.zeros((n_Classifiers, n_Classifiers))
    # Then within the test_dev
    KL_divergence_test_dev_labels = np.zeros((n_Classifiers, n_Classifiers))
    # Then between the training and test_dev
    KL_divergence_training_test_dev_labels = np.zeros((n_Classifiers, n_Classifiers))
    for i in range(n_Classifiers):
        for j in range(n_Classifiers):
            KL_divergence_training_labels[i,j] = KL_divergence(training_nparray_labels[f'{i}_clusters'], training_nparray_labels[f'{j}_clusters'], n_clusters_wanted)
            KL_divergence_test_dev_labels[i,j] = KL_divergence(test_dev_nparray_labels[f'{i}_clusters'], test_dev_nparray_labels[f'{j}_clusters'], n_clusters_wanted)
            KL_divergence_training_test_dev_labels[i,j] = KL_divergence(training_nparray_labels[f'{i}_clusters'], test_dev_nparray_labels[f'{j}_clusters'], n_clusters_wanted)

    # store this data in a dictionary, keys are labelled with n_clusters_wanted, random_seed, n_neighborhoods_wanted, HSV_wanted, weights_startingpoint, gamma_wanted, ScalingChoice_wanted
    key_string = f'n_clusters = {n_clusters_wanted}, random_seed = {random_seed}, n_neighborhoods = {n_neighborhoods_wanted}, HSV = {HSV_wanted}, weights = {weights_startingpoint}, gamma = {gamma_wanted}, ScalingChoice = {ScalingChoice_wanted}'

    MegaDict[f'KL_divergence_Train_vs_test_dev_{key_string}'] = KL_divergence_Train_vs_test_dev
    MegaDict[f'KL_divergence_training_labels_{key_string}'] = KL_divergence_training_labels
    MegaDict[f'KL_divergence_test_dev_labels_{key_string}'] = KL_divergence_test_dev_labels
    MegaDict[f'KL_divergence_training_test_dev_labels_{key_string}'] = KL_divergence_training_test_dev_labels

        
    Healthy_training_l, Healthy_test_dev_l, Disease_training_l, Disease_test_dev_l = get_Healthy_vs_Disease_npArrays(training_nparray_labels = training_nparray_labels, 
                                                                                                                    test_dev_nparray_labels = test_dev_nparray_labels, 
                                                                                                                    n_Classifiers = n_Classifiers, 
                                                                                                                    Healthy_labels = [5, 6], 
                                                                                                                    Disease_labels = [0, 3, 4])

    KL_divergence_train_HealthvsDisease = KL_divergence(Healthy_training_l, Disease_training_l, n_clusters_wanted)
    KL_divergence_test_dev_HealthvsDisease = KL_divergence(Healthy_test_dev_l, Disease_test_dev_l, n_clusters_wanted)
    KL_divergence_healthy_trainvstestdev = KL_divergence(Healthy_training_l, Healthy_test_dev_l, n_clusters_wanted)
    KL_divergence_disease_trainvstestdev = KL_divergence(Disease_training_l, Disease_test_dev_l, n_clusters_wanted)

    MegaDict[f'KL_divergence_train_HealthvsDisease_{key_string}'] = KL_divergence_train_HealthvsDisease
    MegaDict[f'KL_divergence_test_dev_HealthvsDisease_{key_string}'] = KL_divergence_test_dev_HealthvsDisease
    MegaDict[f'KL_divergence_healthy_trainvstestdev_{key_string}'] = KL_divergence_healthy_trainvstestdev
    MegaDict[f'KL_divergence_disease_trainvstestdev_{key_string}'] = KL_divergence_disease_trainvstestdev

    KL_singles = [KL_divergence_Train_vs_test_dev,\
                KL_divergence_train_HealthvsDisease,\
                    KL_divergence_test_dev_HealthvsDisease,\
                        KL_divergence_healthy_trainvstestdev,\
                            KL_divergence_disease_trainvstestdev]

    # save the dict as a pickle with the name f'Megadict_nClus_{n_clusters_wanted}_seed_{random_seed}_nhoods_{n_neighborhoods_wanted}.pkl'
    
    savename = f'Megadict_nClus_{n_clusters_wanted}_seed_{random_seed}_nhoods_{n_neighborhoods_wanted}.pkl'
    if wantSave:
        savename = f'Megadict_nClus_{n_clusters_wanted}_seed_{random_seed}_nhoods_{n_neighborhoods_wanted}.pkl'
        with open(savename, 'wb') as f:
            pickle.dump(MegaDict, f)


    # Create a figure for the KL divergences
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.heatmap(KL_divergence_training_labels, ax=ax[0, 0], cmap=colormap)
    ax[0, 0].set_title('Training: Internal Label Differences')
    sns.heatmap(KL_divergence_test_dev_labels, ax=ax[0, 1], cmap=colormap)
    ax[0, 1].set_title('Test_dev: Internal Label Differences')
    sns.heatmap(KL_divergence_training_test_dev_labels, ax=ax[1, 0], cmap=colormap)
    ax[1, 0].set_title('Training vs Test_dev: Label Differences')
    ax[1, 1].plot(range(5), KL_singles, marker='o')
    # set the xticks to be the names of the KL_singles
    ax[1, 1].set_xticks(range(5))
    ax[1, 1].set_xticklabels(['Train vs Test_dev', 'Train_Health vs Disease', 'Test_dev_Health vs Disease', 'Healthy Train vs Test_dev', 'Disease Train vs Test_dev'])
    ax[1, 1].tick_params(axis='x', rotation=45)
    ax[1, 1].set_title('Single KL divergences')
    plt.suptitle(f'{savename[:-4]}')
    plt.tight_layout()

    # Save the image
    fig.savefig(f'{savename[:-4]}.png')

    time.sleep(1)
    
    plt.close('all')
    
    if annotate:
        print(f'Finished; Timepassed = {time.time()-timer_0:.2f}s')
    
    
    return MegaDict, KMeansclus, scaler_fitted, cols_to_add, key_string

# Make a function that draws the RGB image given the index of the image
def draw_img_from_index(img_idx):
    # Load the image
    csv_ = pd.read_csv(f'{dir}\\\Extracted_and_expanded\\{files[img_idx]}')
    # Take columns 'Y', 'X', 'RGB: FAF' and 'RGB: GREEN' and 'RGB: IR'
    Y_ = csv_['Y']
    X_ = csv_['X']
    RGB_FAF = csv_['RGB: FAF']
    RGB_GREEN = csv_['RGB: GREEN']
    RGB_IR = csv_['RGB: IR']
    # Create the RGB image
    RGB_image = np.zeros((Y_.max()+1, X_.max()+1, 3), dtype=np.uint8)
    RGB_image[:,:,0] = RGB_FAF.values.reshape(RGB_image.shape[0], RGB_image.shape[1])
    RGB_image[:,:,1] = RGB_GREEN.values.reshape(RGB_image.shape[0], RGB_image.shape[1])
    RGB_image[:,:,2] = RGB_IR.values.reshape(RGB_image.shape[0], RGB_image.shape[1])

    # Figure for the RGB image
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(RGB_image)
    ax.set_title(f'RGB image: {files[img_idx]}')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    return csv_, RGB_image


def draw_img_from_csv_(csv_, R, G, B, showimg = False):
    # Create the RGB image
    RGB_image = np.zeros((csv_['Y'].max()+1, csv_['X'].max()+1, 3), dtype=np.uint8)
    RGB_image[:,:,0] = csv_[R].values.reshape(RGB_image.shape[0], RGB_image.shape[1])
    RGB_image[:,:,1] = csv_[G].values.reshape(RGB_image.shape[0], RGB_image.shape[1])
    RGB_image[:,:,2] = csv_[B].values.reshape(RGB_image.shape[0], RGB_image.shape[1])

    if showimg:
        # Figure for the RGB image
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(RGB_image)
        ax.set_title(f'RGB image: {files[img_idx]}')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    
    return RGB_image


def get_weights_from_params(params):
    weight_idx = params.find('weights_')
    gamma_idx = params.find('_gamma')
    str_post_weights = params[weight_idx+len('weights_'):gamma_idx]
    weights = [float(i) for i in str_post_weights.replace('_', '.').split('-')]
    gamma = float(params[gamma_idx+len('_gamma_'):].replace('_', '.'))
    return weights, gamma

# Predict the clusters for the csv_
def predict_cluster_for_csv(img_idx, dir, files, model_params):
    # Extract the relevant columns
    output_dict = pickle.load(open(model_params+'.pkl', 'rb'))
    KMeansclus = output_dict['KMeansclus']
    scaler_fitted = output_dict['scaler_fitted']
    cols_to_add = output_dict['cols_to_add']
    csv_ = pd.read_csv(f'{dir}\\\Extracted_and_expanded\\{files[img_idx]}')
    
    csv_nparray = csv_[cols_to_add].to_numpy()
    
    weight_baseline, gamma = get_weights_from_params(model_params)
    weights = create_weighing_array(n_basecols = len(weight_baseline), 
                                    cols_to_add = cols_to_add, 
                                    weight_baseline = weight_baseline, 
                                    gamma = gamma)
    
    csv_nparray_scaled = scaler_fitted.transform(csv_nparray)
    csv_nparray_scaled_weighted = weightArray(training_nparray = csv_nparray_scaled, weights = weights)
    
    # Predict
    Predicted_Cluster = KMeansclus.predict(csv_nparray_scaled_weighted)
    # add predictions into the csv_ as "Predicted_Cluster"
    csv_['Predicted_Cluster'] = Predicted_Cluster
    
    return csv_

# Use clusters to add contours to the image
def get_cluster_array(csv):
    image_ = np.zeros((csv['Y'].max()+1, csv['X'].max()+1), dtype=np.uint8)
    # insert the predicted clusters into the image
    image_[:,:] = csv['Predicted_Cluster'].values.reshape(image_.shape[0], image_.shape[1])
    return image_


def extract_clusterprofiles(model_params, show_figure = False):
    n_clusters = int(model_params.split('_')[3])
    output_dict = pickle.load(open(model_params+'.pkl', 'rb'))
    training_nparray_labels = output_dict['MegaDict']['training_nparray_labels']
    clusters = np.zeros((8+3, n_clusters))
    if show_figure:
        plt.figure(figsize=(12, 5))
    Labelmeaning = ['RPD', 'Small Drusen', 'Medium Drusen', 'Large Drusen', 'Pigmentary Abnormalities', 'Blood Vessels', 'Background', 'Cuticular Drusen', 'Healthy', 'Disease', 'Random']
    for i in range(8):
        for j in range(n_clusters):
            clusters[i, j] = np.sum(training_nparray_labels[f'{i}_clusters'] == j)/len(training_nparray_labels[f'{i}_clusters'])# indexes 0 to 7 are the labels
        if show_figure:
            plt.plot(range(n_clusters), clusters[i, :], label=f'{Labelmeaning[i]}', alpha=0.7, marker='o', color=f'C{i}')

    for j in range(n_clusters):
        clusters[8, j] = np.sum(output_dict['MegaDict']['Healthy_training_l'] == j)/len(output_dict['MegaDict']['Healthy_training_l'])# index 8 is the healthy
        clusters[9, j] = np.sum(output_dict['MegaDict']['Disease_training_l'] == j)/len(output_dict['MegaDict']['Disease_training_l'])# index 9 is the disease
        clusters[10, j] = np.sum(output_dict['MegaDict']['train_clusters'] == j)/len(output_dict['MegaDict']['train_clusters'])# index 10 is all labels together "random"

    
    if show_figure:
        
        plt.plot(range(n_clusters), clusters[8, :], label=f'{Labelmeaning[8]}', color='green')
        plt.plot(range(n_clusters), clusters[9, :], label=f'{Labelmeaning[9]}', color='red')
        plt.plot(range(n_clusters), clusters[10, :], label=f'{Labelmeaning[10]}', color='black')
        plt.legend()
        plt.title(f'Cluster distribution for n_clusters = {n_clusters}')
        plt.xlabel('Cluster number')
        plt.ylabel('Fraction of pixels')
        plt.tight_layout()
        
    return clusters, Labelmeaning


def get_likelihood_images(clusters, clusters_wanted, image_baseline, n_clusters, jumpsize_L_R_U_D = 4):

    cluster_checks = clusters[clusters_wanted,:]
    image_3 = np.zeros((image_baseline.shape[0], image_baseline.shape[1], len(clusters_wanted)), dtype=np.float64)
    image_euclidean = image_3.copy()
    image_cosine = image_3.copy()
    image_KL = image_3.copy()

    for x_ in range(0+jumpsize_L_R_U_D, image_baseline.shape[0]-jumpsize_L_R_U_D):
        for y_ in range(0+jumpsize_L_R_U_D, image_baseline.shape[1]-jumpsize_L_R_U_D):
            current_square = image_baseline[x_-jumpsize_L_R_U_D:x_+jumpsize_L_R_U_D, y_-jumpsize_L_R_U_D:y_+jumpsize_L_R_U_D]
            current_square_hist = np.array(np.histogram(current_square, bins = range(n_clusters+1))[0], dtype=np.float64)
            # Normalize current_square_hist to make it a distribution
            current_square_hist_normalized = current_square_hist / np.sum(np.array(np.histogram(current_square, bins = range(n_clusters+1))[0], dtype=np.float64))
            image_euclidean[x_, y_,:] , image_cosine[x_, y_,:] , image_KL[x_, y_,:] = get_likelihoods(current_square_hist_normalized, cluster_checks)
    return image_euclidean, image_cosine, image_KL


def draw_hexplot(RGB_image, cluster_img, n_clusters, figsize_wanted = (15, 10)):
    fig, axs = plt.subplots(2, 3, figsize = figsize_wanted)
    # remove the axis
    sns.heatmap(RGB_image[:,:,0], ax=axs[0, 0], cmap='bone', cbar=False, xticklabels=False, yticklabels=False)
    axs[0, 0].set_title('FAF')
    sns.heatmap(RGB_image[:,:,1], ax=axs[0, 1], cmap='bone', cbar=False, xticklabels=False, yticklabels=False)
    axs[0, 1].set_title('Green')
    sns.heatmap(RGB_image[:,:,2], ax=axs[0, 2], cmap='bone', cbar=False, xticklabels=False, yticklabels=False)
    axs[0, 2].set_title('IR')
    # make the heatmap cmap ticks show n_clusters
    cmap = sns.color_palette('Spectral', n_clusters)
    sns.heatmap(cluster_img, ax=axs[1, 0], cbar=True, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, fmt='d', vmin = -0.5, vmax=n_clusters-0.5)
    axs[1, 0].set_title('Predicted Clusters')
    axs[1, 1].imshow(RGB_image)
    axs[1, 1].set_title('RGB image')
    # overlap the two images with 50% transparency
    axs[1, 2].imshow(RGB_image)
    axs[1, 2].imshow(cluster_img, alpha=0.3)
    axs[1, 2].set_title('RGB image with clusters')


def draw_likelihood_hexplot(image, image_2, image_euclidean, image_cosine, image_KL, contour_data, n_clusters, X_r = 0.6, Y_r = 1, n_r = 4):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # plot image and use the same colorbar for all three
    cmap = 'Spectral_r'
    sns.heatmap(image_euclidean, ax=axs[1,0], cbar=True, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, fmt='d', vmin = 0, vmax=1)
    axs[1,0].set_title('Euclidean likelihood')
    sns.heatmap(image_cosine, ax=axs[1,1], cbar=True, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, fmt='d', vmin = 0, vmax=1)
    axs[1,1].set_title('Cosine likelihood')
    sns.heatmap(image_KL, ax=axs[1,2], cbar=True, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, fmt='d', vmin = 0, vmax=1)
    axs[1,2].set_title('KL likelihood')

    cmap = sns.color_palette('gnuplot', n_clusters)
    sns.heatmap(image_2, ax=axs[0, 0], cbar=True, xticklabels=False, yticklabels=False, cmap=cmap, annot=False, fmt='d', vmin = -0.5, vmax=n_clusters-0.5)
    axs[0, 0].set_title('Predicted Clusters')
    axs[0, 1].imshow(image)
    axs[0, 1].set_title('RGB image')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    
    levels = np.linspace(0.6, 1, num=4)  # Example level definition
    contours =  axs[0, 2].contour(contour_data, levels=levels, cmap = 'cool')  # 'alpha' controls the transparency of contours
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')
    axs[0, 2].imshow(image)
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_title('Overlayed likelihood contours from contour_data')
    plt.tight_layout()


