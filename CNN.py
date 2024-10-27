
##############################################################################################################
#### Import Packages 
##############################################################################################################

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns 
import h5py
from torchvision.transforms import v2
import pathlib
import warnings

from tkinter import filedialog
from tkinter import *
import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
#from utils import *
#from model import UNET
import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF 
#from torcheval.metrics.functional.aggregation.auc import auc
from sklearn.metrics import roc_auc_score
import time 


#from torchvision.transforms import v2
from sklearn.utils.class_weight import compute_class_weight

import gc 
import cv2
import random 
import os
from PIL import Image 
import torch
from torch.utils.data import Dataset
#from torchvision import transforms, datasets
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import expit
import pickle 
from datetime import datetime
import GPUtil
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd 
import torch
#from model import UNET
#from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from cityscapesscripts.helpers.labels import trainId2label as t2l
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score
import statistics as stats
from scipy.stats import mode 
import scipy
from skimage.morphology import skeletonize

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
#from datasets import CityscapesDataset
from PIL import Image
from tqdm import tqdm
import numpy as np

from models import *

##############################################################################################################
#### Configuration 
##############################################################################################################
# Enable garbage collection for memory 
gc.enable()

# Removes tkinter popups after selecting directories 
try: 
    root = Tk()
    root.withdraw()
    root.attributes('-topmost',1)
except:
    pass 


class CNN(): 
    def __init__(self, model_name, task, root_directory = None, num_training_epochs = 250, batch_size = 1, number_channels = None, learning_rate = 0.000_1, weight_decay = 0.0):
        """
        This class provides an interface for training convolutional neural network 
        deep learners for semantic segmentation or image classification. 

        Parameters
        ----------
        model_name 
            TYPE:string
            DESCRIPTION:    Optional variable that specifies which architecture to use.
                            Must be a valid option in "list_of_models".
        root_directory
            TYPE:           directory string
            DESCRIPTION:    The topmost directory for storing training data, labels, and trained models. 
                            If not provided, the user will receive a tkinter popup asking for the directory.            
        task 
            TYPE:           string. One of "segmentation" or "classification"
            DESCRIPTION:    Selector for semantic segmentation or image classification. 

        num_training_epochs
            TYPE:           positive integer
            DESCRIPTION:    The number of epochs to train the model. 

        number_channels
            TYPE:           positive integer
            DESCRIPTION:    The number of input image channels. 


        Additional Class States
        ----------
        self.device
            TYPE:           string. One of "cpu" or "cuda:x" where x is 0, 1, 2...
            DESCRIPTION:    The hardware used to train and evaluate models. 
        
        self.learning_rate
            TYPE:           float
            DESCRIPTION:    The initial learning rate for epoch 0. The value typically decreases over training due to schedulers. 
        
        self.weight_decay
            TYPE:           float
            DESCRIPTION:    Regularization term. The exact meaning and effect depends on which optimization algorithm is used. 
        
        self.model
            TYPE:           Pytorch nn.Module
            DESCRIPTION:    The actual model architecture. Not to be confused with self.model_name which is simply a string identifier. 
            
            
            
            
        
        Returns
        -------
        None.

        """
        
        # Ensure a directory is selected, otherwise raise an exception 
        if hasattr(self, 'root_directory') == False:
            self.select_root_directory()
            
        if hasattr(self, 'root_directory') == False:
            raise Exception("Error: A root directory must be selected")
            
        # Set arguments 
        self.task = task
        self.number_channels = number_channels
        self.learning_rate = learning_rate 
        self.weight_decay = weight_decay 
        self.num_training_epochs = num_training_epochs 
        self.transform = None
        self.batch_size = batch_size

        
        # number_channels must be explicitly set
        if (hasattr(self, 'number_channels') == False) or (isinstance(number_channels, int) == False):
            raise Exception("Error: number_channels must be explicitly set to an integer")
        
        # Detect whether Cuda is installed. 
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            print('\nRunning on the GPU')
        else:
            self.device = "cpu"
            print('\nRunning on the CPU')
        
        list_of_models = ["MSU_Net", "ResNet18", "SegNet"]
        
        # If a model path has been provided, load model into memory. 
        if (model_name is not None):
            if (model_name in list_of_models): 
                self.model_name = model_name
                self.instantiate_model()
            else:
                raise Exception("\nError: Model must be specified.")
        
        
        
        self.optimizer = optim.Adam(params = self.model.parameters(), 
                               lr = self.learning_rate,
                               weight_decay = self.weight_decay
                               )
        
        try:
            self.optimizer.load_state_dict(self.checkpoint['optim_state_dict'])
        except:
            self.optimizer.param_groups[0]['lr'] = self.learning_rate
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.gamma = 0.99
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma = self.gamma)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=255)  
        
        #self.initialize_parameters_for_training_from_scratch()
        
        if 'saved_model' in os.listdir( self.model_directory ):
            print("\nDetected previously saved model, loading Now.")
            try:
                self.load_checkpoint() 
                #self.model.load_state_dict(self.checkpoint['model_state_dict'])
                print("\nCompleted Loading Previous Model.")
            except:
                print("\nError: Unable to Load Previous Model.")
         
        print("\nCNN Class Setup Complete")
        
    
    def select_root_directory(self):
        """
        This function prompts the user to select a root directory. 
        Inside the root dir there must be the following sub-directories: 
            raw 
            masks 
            validation 
            predictions 
                
        After selecting the root dir, this function checks that all of the directories 
        do actually exist, and then checks the integrity of the files in the raw, mask, and validation directories. 

        Returns
        -------
        None.

        """
        self.root_directory             = filedialog.askdirectory(title='Select Root Directory') 
        
        self.data_directory             = os.path.join(self.root_directory, 'data')
        self.raw_data_directory         = os.path.join(self.root_directory, 'data', 'raw')
        self.mask_data_directory        = os.path.join(self.root_directory, 'data', 'masks')
        self.validation_data_directory  = os.path.join(self.root_directory, 'data', 'validation')
        self.predicted_data_directory   = os.path.join(self.root_directory, 'data', 'predictions')
        self.model_directory            = os.path.join(self.root_directory, 'model')
        
        directories = [self.root_directory, self.data_directory, self.raw_data_directory, self.mask_data_directory, self.validation_data_directory, self.predicted_data_directory, self.model_directory]
        
        for directory in directories:
            # Verify that the directory exists 
            if os.path.isdir(directory):
                pass
            else:
                os.mkdir(directory)
                print("")
                warnings.warn("Error: Directory " + str(directory) + " was not located. Directory was created.")
                print("")
                
            # Verify the integrity of the files. 
            self.verify_images(directory) 
        
    
    def verify_images(self, directory): 
        """
        

        Parameters
        ----------
        directory : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        image_list = os.listdir(  directory  ) 
        file_already_exists = False
        """
        Make sure we have the state var image_data.
        If not, try loading the CSV from the local machine.
        If the CSV file isn't present, then create an empty dataframe.
        """
        """
        if hasattr(self, 'image_data'):
            pass
        else:
        """
        
        try:
            self.image_data = pd.read_csv(   os.path.join(self.root_directory, 'image_data.csv')   )
            file_already_exists = True
        except:    
            self.image_data = pd.DataFrame() 
    
        # Iterate through all images in the directory. If metadata has already been collected, skip the file. 
        # Otherwise load the file, then record metadata and save when completed. 
        for img_path in tqdm(image_list, desc = str(os.path.basename(directory))): 
            temp = [] 
            
            if 'File Path' in list(   self.image_data.columns   ):
                if os.path.join(directory, img_path) in list(self.image_data['File Path']):
                    continue  
                
            # Load to memory. 
            try:
                if img_path.endswith(('.png', '.jpg', '.tif', '.tiff')):
                    img = np.array( cv2.imread( os.path.join(directory, img_path)), dtype = np.uint8 )
                elif img_path.endswith('.npy'):
                    img = np.load(os.path.join(directory, img_path))
            
                # Record metadata 
                temp.append(os.path.join(directory, img_path))
                temp.append(os.path.basename(directory))
                temp.append(img_path)
                temp.append(img.shape[1])
                temp.append(img.shape[0]) 
                if len(img.shape) == 2:
                    temp.append(1)
                else:
                    pass 
                    temp.append(img.shape[2])
                
                temp = pd.Series(temp, index = ['File Path', 'Directory','Image Name', 'Width (px)', 'Height (px)', 'Number Channels']) 
                
                
                self.image_data = pd.concat([self.image_data, temp.to_frame().T], axis = 0)
                self.image_data.reset_index(inplace=True, drop = True)
                    
            except:
                pass 
        
        
        if file_already_exists:
            pass
        else:
            self.image_data['Designation'] = None
            self.image_data['Encoded Labels'] = None 
            
        self.image_data.drop_duplicates(inplace=True)
        output_path = os.path.join( self.root_directory, 'image_data.csv' )
        self.image_data.to_csv(output_path, index = False)
 
    
        
    def instantiate_model(self): 
        """
        

        Returns
        -------
        None.

        """
        print("\nInitializing Model...")
        if self.model_name == "ResNet18":
            self.model = ResNet(img_channels = self.number_channels, num_layers = 18, block = BasicBlock ).to(self.device)
        elif self.model_name == "MSU_Net":
            self.model = MSU_Net(img_ch=self.number_channels, output_ch=35).to(self.device)
        elif self.model_name == "SegNet":
            self.model = SegNet(in_channels=self.number_channels, out_channels=100, features=64)
        
        
        print("\nModel Initialized")

                
                
    def predict_local_images(self):
        """
        This function takes the files in the "raw" data directory and makes predictions. 
        

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        
        Semantic segmentations are saved as PNG files in the "predictions" directory 
        and image classifications are saved in the save directory as a CSV file. 

        """        
        
        # Set model to eval to shutoff dropout layers and turn of autograd 
        #self.model.eval()
        
        # Get dataloader and file paths. 
        file_names, data = self.get_production_data()
        index = 0
        
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data), total = len(file_names)):
                # Load data. There is no y as these are predictions. 
                X = batch 
                
                if self.task == 'segmentation':
                    # Make sure that the tensor has the correct dimensions, even if they're empty placeholders. 
                    if len(X.shape) == 3:
                        X = X.unsqueeze(0)
                
                # Send batch to correct hardware and make predictions. 
                X = X.to(self.device)
                predictions = self.model(X) 
                
                #print(predictions)
                
                # Take the predictions tensor and convert to encoded integer classifications. 
                pred_labels = torch.nn.functional.softmax(predictions, dim=1)
                pred_labels = torch.argmax(pred_labels, dim=1) 
                #pred_labels = pred_labels.type(torch.uint8)
                
                #print(pred_labels)
                
                #if idx == 1:
                #    break 
                
                if self.task == 'segmentation':
                    # Convert to PIL image and then to NP array. 
                    tensor_pred = transforms.ToPILImage()(pred_labels.byte())
                    tensor_pred = np.array( tensor_pred )
                    
                    # Visualize result with the tab10 mapping. 
                    # The default cmap gives a continuous color map, while we want discrete color maps 
                    # for semantic segmentations - hence the offsets. 
                    cmap = plt.get_cmap('tab10', np.max(tensor_pred) - np.min(tensor_pred) + 1)
    
                    plot_image = plt.imshow(tensor_pred, cmap=cmap, vmin=np.min(tensor_pred) - 0.5, 
                                      vmax=np.max(tensor_pred) + 0.5)
                    
                    # Set the colorbar to tick at integers. 
                    cax = plt.colorbar(plot_image, ticks=np.arange(np.min(tensor_pred), np.max(tensor_pred) + 1))
                    
                    # Turn off ticks at borders. 
                    plt.xticks([])
                    plt.yticks([])
                    
                    # Set title and display the image. 
                    plt.title("Example Segmented Image\n" + str(file_names[idx]))
                    plt.show() 
                    
                    # Now save the predictions as well. 
                    # First get the base file name so we can reuse it. 
                    basename = os.path.splitext(   os.path.basename(file_names[idx])   )[0]
                    
                    # Then calculate the absolute filepath to save to. 
                    # Always save as PNG to avoid loss, for example like in JPG. 
                    filename = os.path.join(self.predicted_data_directory, basename) + '.png'
                    
                    # And save results as png. 
                    cv2.imwrite(filename, tensor_pred)
                    
                elif self.task == 'classification':
                    
                    pred_labels = pred_labels.detach().cpu().numpy()
                    pred_labels = list(pred_labels)
                    
                    for value in pred_labels:
                    
                        self.image_data.at[   index, 'Encoded Prediction'   ] = value
                        index = index + 1
                    """
                    basename = os.path.splitext(   os.path.basename(file_names[idx])   )[0]
                    temp_df = pd.DataFrame( tensor_pred, basename , ignore_index=True)
                    
                    output = pd.concat([output, temp_df], axis = 0)
                    """
                # Clear CUDA memory.
                try:
                    torch.cuda.empty_cache() 
                except:
                    pass 
                
        y_true = self.image_data['Encoded Labels']
        y_pred = self.image_data['Encoded Prediction']
        score = balanced_accuracy_score(y_true, y_pred)
        print("Balanced Accuracy: " + str(score) )
        
        # If classifying images, save results as a CSV. 
        if self.task == 'classification':
            output_path = os.path.join( self.root_directory, 'image_data.csv' )
            self.image_data.to_csv(output_path, index = False)


    def predict_image(self, image):
        """
        

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Set model to eval to shutoff dropout layers and turn of autograd 
        self.model.eval()
        
        if self.task == 'segmentation':
            X = transforms.ToPILImage()(image).to(self.device)
            
            predictions = self.model(X) 
            
            pred_labels = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(pred_labels, dim=1) 
            pred_labels = pred_labels.type(torch.uint8)
            pred_labels = transforms.ToPILImage()(pred_labels.byte())
                
            return pred_labels, predictions
      
    def get_production_data(self):
        """
        

        Parameters
        ----------
        batch_size : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        
        X = [os.path.join( self.raw_data_directory, x) for x in os.listdir(self.raw_data_directory)]
        data = Dataset(
            X,
            y=None,
            transform=self.transform, 
            task='production',
            number_channels = self.number_channels
            )
    
        data_loaded = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False, drop_last=False)
    
        return X, data_loaded        
      
        
    def get_training_data(self, loader_task, test_size = 0.2, percent_annotated = 0.2):
        """
        

        Parameters
        ----------

        eval : TYPE, optional
            DESCRIPTION. The default is False.
        percent_annotated : TYPE, float
            DESCRIPTION. The percentage of mask image that must be annotated for semantic segmentation training. 

        Returns
        -------
        data_loaded : TYPE
            DESCRIPTION.

        """
        if (test_size <= 0) or (test_size >= 1):
            raise Exception("Error: Test size must be greater than 0 and less than 1")
            
        if hasattr(self, 'X') & hasattr(self, 'y'):
            pass 
        else: 
            # If the task is semantic segmentation, make a list of paired masks for the dependent variable.
            # And check that the files have been sufficiently annotated. 
            if self.task == "segmentation":
                
                training_X = self.image_data[self.image_data['Directory'] == 'raw']   
                validation_X = self.image_data[self.image_data['Directory'] == 'validation']   
                y = self.image_data[self.image_data['Directory'] == 'masks']   
                
                #X = [os.path.join( self.raw_data_directory, x) for x in os.listdir(self.raw_data_directory)]
                #y = [os.path.join( self.mask_data_directory, x) for x in os.listdir(self.mask_data_directory)]
                
                self.X = [] # State variables to make tracking easier. 
                self.y = [] 
                self.validation_y = []
                self.validation_X = [] 
                for i , row in y.iterrows():
                    file = row['File Path']
                   
                    # Load to memory. 
                    if file.endswith(('.png', '.jpg', '.tif', '.tiff')):
                        img = np.array( cv2.imread( file ), dtype = np.uint8 )
                    elif file.endswith('.npy'):
                        img = np.load( file )
                    
                    # Check that enough of the image is annotated. 
                    # 255 is unlabled / unannotated training data
                    sufficient_data = float(img[img != 255].size / img.size) >= percent_annotated
                    paired_training_file = row['Image Name'] in list(training_X['Image Name'])
                    paired_validation_file = row['Image Name'] in list(validation_X['Image Name'])
                    
                    if (sufficient_data) & (paired_training_file): 
                        self.y.append( os.path.join( self.mask_data_directory, row['Image Name'])    )
                        self.X.append( os.path.join( self.raw_data_directory, row['Image Name'])    ) 
                        
                    elif (sufficient_data) & (paired_validation_file): 
                        self.validation_y.append(   os.path.join( self.mask_data_directory, row['Image Name'])    )
                        self.validation_X.append(   os.path.join( self.validation_data_directory, row['Image Name'])) 
                        
                        
            # If the task is image classificaiton, load a CSV file that specifies image filepaths (X) in 
            # one column and the encoded classifications (y) in another column. 
            elif self.task == "classification":
                
                # State variables to make tracking easier. 
                self.X = [] 
                self.y = [] 
                self.validation_y = []
                self.validation_X = [] 
                
                path = os.path.join(self.root_directory, "image_data.csv")
                df = pd.read_csv(path)
                
                self.X = list(   df[ df['Designation'] == 'training']['File Path']   )
                self.y = list(   df[ df['Designation'] == 'training']['Encoded Labels']   )
                
                self.validation_X = list(   df[ df['Designation'] == 'validation']['File Path']   )
                self.validation_y = list(   df[ df['Designation'] == 'validation']['Encoded Labels']   )
                
            
        if self.task == 'segmentation':
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_size, shuffle = True)

        elif self.task == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_size, shuffle = True, stratify=self.y)
        
        
            try:
                # We want to be able to retroactively view how many instances were used for 
                # training, testing, and validation. And also view how the instances were distributed 
                # across classes. 
                a = np.array(np.unique(y_train, return_counts = True))
                b = np.array(np.unique(y_test, return_counts = True))
                c = np.array(np.unique(self.validation_y, return_counts = True))
                for encoded_class in np.unique(self.y):
                    index = np.argwhere(a[0,:] == encoded_class )
                    counts = int(a[1, index])
                    
                    # Create the dictionary class if necessary. 
                    if encoded_class in self.training_counts.keys(): 
                        pass
                    else:
                        self.training_counts[encoded_class] = [] 
                    
                    # For each encoded class, append the number of examples used for training each epoch. 
                    self.training_counts[encoded_class].append(counts)
                    ####
                    
                    index = np.argwhere(b[0,:] == encoded_class )
                    counts = int(b[1, index])
                    
                    # Create the dictionary class if necessary. 
                    if encoded_class in self.testing_counts.keys(): 
                        pass
                    else:
                        self.testing_counts[encoded_class] = [] 
                    
                    # For each testing_counts class, append the number of examples used for training each epoch. 
                    self.testing_counts[encoded_class].append(counts)
                    ####
                    
                    index = np.argwhere(c[0,:] == encoded_class )
                    counts = int(c[1, index])
                    
                    # Create the dictionary class if necessary. 
                    if encoded_class in self.validation_counts.keys(): 
                        pass
                    else:
                        self.validation_counts[encoded_class] = [] 
                    
                    # For each encoded class, append the number of examples used for training each epoch. 
                    self.validation_counts[encoded_class].append(counts)
                    ####
            except:
                pass 
                    
        training_data = Dataset(
            X_train,
            y_train,
            transform=self.transform, 
            task=loader_task,
            number_channels = self.number_channels
            )
    
        training_data_loaded = torch.utils.data.DataLoader(training_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
        testing_data = Dataset(
            X_test,
            y_test,
            transform=self.transform, 
            task=loader_task,
            number_channels = self.number_channels
            )
    
        testing_data_loaded = torch.utils.data.DataLoader(testing_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
        try: 
                
            validation_data = Dataset(
                self.validation_X,
                self.validation_y,
                transform=self.transform, 
                task=loader_task,
                number_channels = self.number_channels
                )
            
            validation_data_loaded = torch.utils.data.DataLoader(validation_data, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
            return training_data_loaded, testing_data_loaded, validation_data_loaded
        
        except:
            return training_data_loaded, testing_data_loaded, None
       
           
    def load_checkpoint(self):
        """
        This function loads the most recent saved training checkpoint 
        from the "saved_model" file. 

        Returns
        -------
        None.

        """
        self.checkpoint = torch.load(os.path.join(self.model_directory, 'saved_model')  , map_location='cpu')
        
        self.epoch =                self.checkpoint['epoch']+1
        self.model.state_dict =     self.checkpoint['model_state_dict']
        self.optimizer.state_dict = self.checkpoint['optim_state_dict']
        
        self.training_loss =        self.checkpoint['training_loss_values']
        self.training_accuracy =    self.checkpoint['training_accuracy_values']
        self.training_AUC =         self.checkpoint['training_AUC_values']
        
        self.testing_loss =         self.checkpoint['testing_loss_values']
        self.testing_accuracy =     self.checkpoint['testing_accuracy_values']
        self.testing_AUC =          self.checkpoint['testing_AUC_values']
        
        self.validation_loss =      self.checkpoint['validation_loss_values']
        self.validation_accuracy =  self.checkpoint['validation_accuracy_values']
        self.validation_AUC =       self.checkpoint['validation_AUC_values']
        
        self.training_times =       self.checkpoint['training_times_values']
        self.gpu_temperatures_c =   self.checkpoint['gpu_temperatures']
        self.learning_rates =       self.checkpoint['learning_rates']
        
        self.training_counts =      self.checkpoint['training_counts']
        self.testing_counts =       self.checkpoint['testing_counts']
        self.validation_counts =    self.checkpoint['validation_counts']
        
        
        
    def plot_losses(self):
        """
        This function plots multiple graphs for the training loss, GPU temperature, accuracy, 
        and other parameters. 
        

        Returns
        -------
        None.

        """
        # Load saved torch file and update state variables. 
        self.load_checkpoint() 

        # X axis placeholder for tick marks. 
        epoch_list = list(range(len(self.training_loss)))
        
        fig, axs = plt.subplots(4, 2, figsize=(17, 20), dpi = 450)
        try:
            axs[0,0].plot(epoch_list, self.training_AUC, label = "Training AUC")
            axs[0,0].plot(epoch_list, self.testing_AUC, label = "Testing AUC")
            axs[0,0].plot(epoch_list, self.validation_AUC, label = "Validation AUC")
            axs[0,0].legend(loc='best')
            axs[0,0].set_xlabel('Epochs')
            axs[0,0].set_ylabel('AUC')
            axs[0,0].title.set_text(f"AUC over {self.epoch} epoch/s")
            axs[0,0].grid(True)
        except:
            pass 
        
        try:
            axs[0,1].plot(epoch_list, self.training_accuracy, label = "Training Accuracy")
            axs[0,1].plot(epoch_list, self.testing_accuracy, label = "Testing Accuracy")
            axs[0,1].plot(epoch_list, self.validation_accuracy, label = "Validation Accuracy")
            axs[0,1].legend(loc='best')
            axs[0,1].set_xlabel('Epochs')
            axs[0,1].set_ylabel('Accuracy (%)')
            axs[0,1].title.set_text(f"Accuracy over {self.epoch} epoch/s")
            axs[0,1].grid(True)
        except:
            pass 
        
        try:
            axs[1,0].plot(epoch_list, self.training_loss, label = "Training Loss")
            axs[1,0].plot(epoch_list, self.testing_loss, label = "Testing Loss")
            axs[1,0].plot(epoch_list, self.validation_loss, label = "Validation Loss")
            axs[1,0].legend(loc='best')
            axs[1,0].set_xlabel('Epochs')
            axs[1,0].set_ylabel('Loss')
            axs[1,0].title.set_text(f"Loss over {self.epoch} epoch/s")
            axs[1,0].set_yscale("log")
            axs[1,0].grid(True)
        except:
            pass 
        
        try:
            axs[1,1].plot(epoch_list, self.learning_rates)
            axs[1,1].set_xlabel('Epochs')
            axs[1,1].set_ylabel('Learning Rate')
            axs[1,1].title.set_text(f"Learning Rate over {self.epoch} epoch/s")
            axs[1,1].grid(True)
        except:
            pass 
            
        try:
            axs[2,0].plot(epoch_list, [t.total_seconds() for t in self.training_times] )
            axs[2,0].set_xlabel('Epochs')
            axs[2,0].set_ylabel('Training Time (s)')
            axs[2,0].title.set_text(f"Training Time over {self.epoch} epoch/s")
            axs[2,0].grid(True)
        except:
            pass 
        
        try:
            axs[2,1].plot(epoch_list, self.gpu_temperatures_c)
            axs[2,1].set_xlabel('Epochs')
            axs[2,1].set_ylabel('GPU Temperature (C)')
            axs[2,1].title.set_text(f"GPU Temperature over {self.epoch} epoch/s")
            axs[2,1].grid(True)
        except:
            pass 
        
        try:
            for encoded_class in self.training_counts.keys():             
                axs[3,0].plot(self.training_counts[encoded_class], label = "Training Counts - Class " + str(encoded_class))
            for encoded_class in self.testing_counts.keys():             
                axs[3,0].plot(self.testing_counts[encoded_class], label = "Testing Counts - Class " + str(encoded_class))
            for encoded_class in self.validation_counts.keys():             
                axs[3,0].plot(self.validation_counts[encoded_class], label = "Validation Counts - Class " + str(encoded_class))
            axs[3,0].set_xlabel('Epochs')
            axs[3,0].set_ylabel('Instances (counts)')
            axs[3,0].title.set_text(f"Instances over {self.epoch} epoch/s")
            axs[3,0].legend(loc='best')
            axs[3,0].grid(True)
        except:
            pass 
        
        plt.show() 
            
           
    def initialize_parameters_for_training_from_scratch(self):
        """
        This function resets all training parameters other than the model state and optimizer state. 

        Returns
        -------
        None.

        """
        # epoch is initially assigned to 0. If LOAD_MODEL is true then
        # epoch is set to the last value + 1. 
        self.epoch = 0 
        
        self.training_loss = []
        self.training_accuracy = [] 
        self.training_AUC = [] 
        
        self.testing_loss = [] 
        self.testing_accuracy = [] 
        self.testing_AUC = [] 

        self.validation_loss = [] 
        self.validation_accuracy = [] 
        self.validation_AUC = [] 
        
        self.training_times = []
        self.gpu_temperatures_c = []
        self.learning_rates = []
        
        self.training_counts = {}   
        self.testing_counts = {}   
        self.validation_counts = {}   
        
        
    def reset_training_log(self): 
        """
        This function is used to reset the training log of a model, and leaves only the 
        model state and optimizer state as their original values. 

        Returns
        -------
        None.

        """
        
        try:
            self.load_checkpoint()
        except:
            pass
        
        try:
            self.initialize_parameters_for_training_from_scratch() 
        except:
            pass 
        
        
        torch.save({
            'model_state_dict': self.model.state_dict,
            'optim_state_dict': self.optimizer.state_dict,
            'epoch': self.epoch,
            
            'training_loss_values': self.training_loss,
            'training_accuracy_values': self.training_accuracy, 
            'training_AUC_values': self.training_AUC,
            
            'testing_loss_values': self.testing_loss,
            'testing_accuracy_values': self.testing_accuracy, 
            'testing_AUC_values': self.testing_AUC,
            
            'validation_loss_values': self.validation_loss,
            'validation_accuracy_values': self.validation_accuracy, 
            'validation_AUC_values': self.validation_AUC,
            
            'training_times_values': self.training_times,
            'gpu_temperatures': self.gpu_temperatures_c,
            'learning_rates': self.learning_rates,
            
            'training_counts': self.training_counts,
            'testing_counts': self.testing_counts,
            'validation_counts': self.validation_counts   
            
        }, os.path.join(self.root_directory, 'model', 'saved_model' ))
        
        
        
        
        
    def train_model(self):
        """
        

        Returns
        -------
        None.

        """
        
        torch.set_flush_denormal(True)
        
        # Try loading a previously saved model to continue training. 
        if 'saved_model' in os.listdir( self.model_directory ):
            print("\nDetected previously saved model, loading now.")
            try:
                self.load_checkpoint() 
                print("\nCompleted Loading Previous Model.")
                
                # Step the scheduler to account for previous training. 
                # Otherwise the learning rate will be incorrect. 
                if self.epoch != 0:
                    for e in range(self.epoch):
                        self.scheduler.step()
                        
            except:
                print("\nError: Unable to Load Previous Model.")
                self.initialize_parameters_for_training_from_scratch()
        else:
            self.initialize_parameters_for_training_from_scratch() 
              
        #Training the model for every epoch. 
        for e in range(self.epoch, self.num_training_epochs):
            # Record the startint time to keep tract of computational expense.
            start_time = datetime.now() 
            
            print(f'Epoch: {e}')
            # Clear memory each epoch to reduce chances of a crash. 
            torch.cuda.empty_cache()
            gc.collect()
            
            # Make a test - train split for each epoch. 
            training_data, testing_data, validation_data = self.get_training_data(test_size = 0.3, percent_annotated=0.2, loader_task=self.task) 
            
            # We will record the loss for each training batch and then calculate an average loss for the 
            # epoch for record keeping purposes. 
            epoch_loss_values = []
            epoch_accuracy_values = [] 
            epoch_AUC_values = []
            
            #### Training 
            for index, batch in enumerate(tqdm(training_data)): 
                X, y = batch
                X = X.float()
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Calculate the probability matrix for each class 
                preds = self.model(X)
                
                #print(preds)
                
                # Find the most likely class. 
                predicted_labels = torch.nn.functional.softmax(preds, dim=1)
                predicted_labels = torch.argmax(predicted_labels, dim=1) 
                predicted_labels = predicted_labels.type(torch.uint8)
                
                epoch_accuracy_values.append(   balanced_accuracy_score(y[y!=255].detach().cpu().numpy(), predicted_labels[y!=255].detach().cpu().numpy())   )
                #epoch_accuracy_values.append(   accuracy_score(y[y!=255].detach().cpu().numpy(), predicted_labels[y!=255].detach().cpu().numpy())   )
                #epoch_accuracy_values.append(   (torch.sum(   (y == predicted_labels)&(y!=255)   ) / torch.sum(y!=255)).detach().cpu().numpy()   )
                try:
                    epoch_AUC_values.append(   roc_auc_score(   y[y!=255].flatten().detach().cpu(), preds[y!=255].flatten().detach().cpu()    )   )
                except:
                    pass 
                
                preds = preds.to( self.device, dtype = torch.float)
                preds.requires_grad_()   
                y = y.to(self.device, dtype = torch.long)

                self.optimizer.zero_grad()
                loss = self.loss_function(preds, y)
                loss.backward()
                epoch_loss_values.append(loss.detach().cpu().numpy())
                self.optimizer.step()
                
                
            
            self.training_loss.append(   np.mean(epoch_loss_values)   )
            self.training_accuracy.append(   np.mean(epoch_accuracy_values)   )
            self.training_AUC.append(   np.mean(epoch_AUC_values)   )
            
            
            #### Testing
            epoch_loss_values = []
            epoch_accuracy_values = [] 
            epoch_AUC_values = []
            for index, batch in enumerate(tqdm(testing_data)): 
                X, y = batch
                X = X.float()
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Calculate the probability matrix for each class 
                preds = self.model(X)
                
                # Find the most likely class. 
                predicted_labels = torch.nn.functional.softmax(preds, dim=1)
                predicted_labels = torch.argmax(predicted_labels, dim=1) 
                predicted_labels = predicted_labels.type(torch.uint8)
                
                epoch_accuracy_values.append(   balanced_accuracy_score(y[y!=255].detach().cpu().numpy(), predicted_labels[y!=255].detach().cpu().numpy())   )

                #epoch_accuracy_values.append(   (torch.sum(   (y == predicted_labels)&(y!=255)   ) / torch.sum(y!=255)).detach().cpu().numpy()   )
                try:
                    epoch_AUC_values.append(   roc_auc_score(   y[y!=255].flatten().detach().cpu(), preds[y!=255].flatten().detach().cpu()    )   )
                except:
                    pass 
                
                
                    
                preds = preds.to( self.device, dtype = torch.float)
                preds.requires_grad_()   
                y = y.to(self.device, dtype = torch.long)
                
                #self.optimizer.zero_grad()
                loss = self.loss_function(preds, y)
                #loss.backward()
                epoch_loss_values.append(loss.detach().cpu().numpy())
                #self.optimizer.step()
                
            self.testing_loss.append(   np.mean(epoch_loss_values)   )
            self.testing_accuracy.append(   np.mean(epoch_accuracy_values)   )
            self.testing_AUC.append(   np.mean(epoch_AUC_values)   )
            

            #### Validation
            epoch_loss_values = []
            epoch_accuracy_values = [] 
            epoch_AUC_values = []
            
            epoch_predicted_scores = []
            epoch_predicted_labels = [] 
            
            for index, batch in enumerate(tqdm(validation_data)): 
                X, y = batch
                X = X.float()
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Calculate the probability matrix for each class 
                preds = self.model(X)
                
                # Find the most likely class. 
                predicted_labels = torch.nn.functional.softmax(preds, dim=1)
                predicted_labels = torch.argmax(predicted_labels, dim=1) 
                predicted_labels = predicted_labels.type(torch.uint8)
                
                epoch_accuracy_values.append(   balanced_accuracy_score(y[y!=255].detach().cpu().numpy(), predicted_labels[y!=255].detach().cpu().numpy())   )

                #epoch_accuracy_values.append(   (torch.sum(   (y == predicted_labels)&(y!=255)   ) / torch.sum(y!=255)).detach().cpu().numpy()   )
                try:
                    epoch_AUC_values.append(   roc_auc_score(   y[y!=255].flatten().detach().cpu(), preds[y!=255].flatten().detach().cpu()    )   )
                except:
                    pass 
                
                
                preds = preds.to( self.device, dtype = torch.float)
                preds.requires_grad_()   
                y = y.to(self.device, dtype = torch.long)
                
                #self.optimizer.zero_grad()
                loss = self.loss_function(preds, y)
                #loss.backward()
                epoch_loss_values.append(loss.detach().cpu().numpy())
                #self.optimizer.step()
            
            self.validation_loss.append(   np.mean(epoch_loss_values)   )
            self.validation_accuracy.append(   np.mean(epoch_accuracy_values)   )
            self.validation_AUC.append(   np.mean(epoch_AUC_values)   )
            
            
            try:
                gpu = GPUtil.getGPUs()[0]
                self.gpu_temperatures_c.append(gpu.temperature)
            except:
                pass 
            
            learning_rate = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(learning_rate)
            
            end_time = datetime.now()
            self.training_times.append(end_time - start_time)
            
            # Step the training rate scheduler.
            self.scheduler.step()
            
            #### Save epoch data 
            
            torch.save({
                'model_state_dict': self.model.state_dict,
                'optim_state_dict': self.optimizer.state_dict,
                'epoch': e,
                
                'training_loss_values': self.training_loss,
                'training_accuracy_values': self.training_accuracy, 
                'training_AUC_values': self.training_AUC,
                
                'testing_loss_values': self.testing_loss,
                'testing_accuracy_values': self.testing_accuracy, 
                'testing_AUC_values': self.testing_AUC,
                
                'validation_loss_values': self.validation_loss,
                'validation_accuracy_values': self.validation_accuracy, 
                'validation_AUC_values': self.validation_AUC,
                
                'training_times_values': self.training_times,
                'gpu_temperatures': self.gpu_temperatures_c,
                'learning_rates': self.learning_rates,
                
                'training_counts': self.training_counts,
                'testing_counts': self.testing_counts,
                'validation_counts': self.validation_counts   
                
            }, os.path.join(self.root_directory, 'model', 'saved_model' ))
        

            # Save model every X epochs. Saving every epoch can quickly result in harddrive memory depletion. 
            if (e % 5) == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict,
                    'optim_state_dict': self.optimizer.state_dict,
                    'epoch': e,
                    
                    'training_loss_values': self.training_loss,
                    'training_accuracy_values': self.training_accuracy, 
                    'training_AUC_values': self.training_AUC,
                    
                    'testing_loss_values': self.testing_loss,
                    'testing_accuracy_values': self.testing_accuracy, 
                    'testing_AUC_values': self.testing_AUC,
                    
                    'validation_loss_values': self.validation_loss,
                    'validation_accuracy_values': self.validation_accuracy, 
                    'validation_AUC_values': self.validation_AUC,
                    
                    'training_times_values': self.training_times,
                    'gpu_temperatures': self.gpu_temperatures_c,
                    'learning_rates': self.learning_rates,
                    
                    'training_counts': self.training_counts,
                    'testing_counts': self.testing_counts,
                    'validation_counts': self.validation_counts  
                }, os.path.join(self.root_directory, 'model', 'saved_model_epoch ' + str(e) ))
                
                
                for channel in range(X.shape[1]):
                    plt.imshow(X[0,channel, :,:].detach().cpu().numpy())
                    plt.title("Channel: " + str(channel) )
                    plt.show() 
                    
                    
            # If this iteration has a lower testing loss than all previous epochs, save as the best model. 
            if np.mean(epoch_loss_values) < np.min(self.testing_loss):
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'epoch': e,
                    
                    'training_loss_values': self.training_loss,
                    'training_accuracy_values': self.training_accuracy, 
                    'training_AUC_values': self.training_AUC,
                    
                    'testing_loss_values': self.testing_loss,
                    'testing_accuracy_values': self.testing_accuracy, 
                    'testing_AUC_values': self.testing_AUC,
                    
                    'validation_loss_values': self.validation_loss,
                    'validation_accuracy_values': self.validation_accuracy, 
                    'validation_AUC_values': self.validation_AUC,
                    
                    'training_times_values': self.training_times,
                    'gpu_temperatures': self.gpu_temperatures_c,
                    'learning_rates': self.learning_rates,
                    
                    'training_counts': self.training_counts,
                    'testing_counts': self.testing_counts,
                    'validation_counts': self.validation_counts  
                }, os.path.join(self.root_directory, 'model', 'best_saved_model' ))
                
                
            self.plot_losses()
                
                
            
            
            
    def copy_annotated_images_from_NPY(self):
    
        # Get user to select the directory with annotated files.
        hdf5_directory = filedialog.askdirectory(title='Select Directory With NPY Files') 
    
        # Make a list of the files of interest. 
        list_annotated_files = list(pathlib.Path(   str(hdf5_directory)   ).glob('*annotation.npy'))
        
        # Load each annotation image and save as PNG.
        for file_path in list_annotated_files:
            file_path = str(file_path)
            if file_path.endswith('.npy'):
                data = np.load(file_path)
                data = data[:,:,-1]
                data = np.array(data, dtype = 'uint8')
                
                basename = os.path.splitext(   os.path.basename(file_path)   )[0]
                basename = basename.replace('.png_annotation', '')
                output_path = os.path.join(self.mask_data_directory, basename) + '.png'
                cv2.imwrite(output_path, data)
        
    
        
                
                
                
                
                
                
        
        
class Dataset(Dataset):
    def __init__(self, X = None, y = None, transform=None, task = False, number_channels = None):
        """
        

        Parameters
        ----------
        X : TYPE, optional
            DESCRIPTION. The default is None.
        y : TYPE, optional
            DESCRIPTION. The default is None.
        transform : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.transform = transform
        self.number_channels = number_channels
        """
        if len(y) == len(X):
            pass 
        else:
            raise Exception("\nError: Lists of X and y must have the same length.")
        """
    
        self.X = X
        self.y = y 
        self.task = task 
    
                
    def __len__(self):
        return len(self.X)
      
    def __getitem__(self, index):
        
        if self.X[index].endswith(('.png', '.jpg', '.tif', '.tiff')):
            image = Image.open(self.X[index])
            if self.number_channels == 1:
                image = transforms.Grayscale()(image)
                #image = image.expand(self.number_channels,*image.shape[1:])
            image = transforms.ToTensor()(image)    
            
            
        elif self.X[index].endswith((".hdf5")):
            with h5py.File( self.X[index] , "r") as f:
                #print(f.keys())
                #print(f['scan01'].keys() )
                #img = f['scan01']['sectionImage'][...]
                
                #img_dims = (496, 1024)
                #img_dims = (1050, 1400)
                
                #image = np.zeros(shape = (1400, 1050, 9))
                #image = torch.zeros( 9, img_dims[0], img_dims[1] )
                image = torch.zeros( self.number_channels, img_dims[0], img_dims[1] )
                """
                for i, key in enumerate(f['ETDRS Measurements'].keys()):
                    temp = f['ETDRS Measurements'][key][...]  
                    temp = transforms.ToTensor()(temp)
                    if (temp.shape[1] != img_dims[0]) or (temp.shape[2] != img_dims[1]):
                        temp = v2.Resize(size=(img_dims[0],img_dims[1]), antialias=True)(temp)
                    image[i,:,:] = temp
                """   
                    
                for key in f.keys():
                    if 'scan' in key:
                        i = int(key.replace('scan', ''))-1
                        temp = f[key]['sectionImage'][...]  
                        temp = transforms.ToTensor()(temp)
                        if (temp.shape[1] != img_dims[0]) or (temp.shape[2] != img_dims[1]):
                            temp = v2.Resize(size=(img_dims[0],img_dims[1]), antialias=True)(temp)
                        image[i,:,:] = temp
                
                        
                    
            #image = np.array(image) 
            #image = image.reshape(image.shape[2], image.shape[1], image.shape[0])
            
            #plt.imshow(image[:,:,4]) 
            #image = transforms.ToTensor()(image)      
            #image = np.reshape(image, newshape = (image.shape[1], image.shape[0], image.shape[2] ) )
            #image = v2.Resize(size=(1400,1050), antialias=True)(image)
        img_dims = (1536, 1536)
        """
        img_dims = (1536, 1536)
            
        if (image.shape[1] != img_dims[0]) or (image.shape[2] != img_dims[1]):
            image = v2.Resize(size=(img_dims[0],img_dims[1]), antialias=True)(image)
        """
        
        
        if self.task == "segmentation":
            #image = Image.open(self.X[index])
            #image = transforms.ToTensor()(image)    
            #image = image.expand(self.number_channels,*image.shape[1:])
            
            
            y = Image.open(self.y[index])
            #y = transforms.ToTensor()(y)    
            y = np.array(y)
            
            
            if (y.shape[0] != img_dims[0]) or (y.shape[1] != img_dims[1]):
                #y = v2.Resize(size=(img_dims[0],img_dims[1]), antialias=True)(y)
                y = cv2.resize(y, dsize=img_dims, interpolation=cv2.INTER_CUBIC)
                
            y = torch.from_numpy(y)
            y = y.type(torch.uint8)
            
                          
        elif self.task == "classification":
            y = self.y[index]
            y = np.array(y)
            y = torch.from_numpy(y)
            y = y.type(torch.float)
            
        elif self.task == "production":
            return image 
            #y = None
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, y        
        
        
def check_GPU_temperature(maximum_temperature_C):
    """
    This function checks the GPU temperature and gives a delay if it is over a threshold, 
    in order to allow it to cool down. 

    Returns
    -------
    None.

    """
    
 
    try:
        temperature_C = GPUtil.getGPUs()[0].temperature
    except:
        raise Warning("No GPU Detected")
    
    while temperature_C > maximum_temperature_C:
        
        # Wait X seconds to cool down. 
        time.sleep(0.1) 

        # Measure temperature again.
        temperature_C = GPUtil.getGPUs()[0].temperature
        
    return None 



    












    

def score(MASK_DATA_DIR, PRED_DATA_DIR): 
    
    predictions = []
    ground_truth = []
    
    for image_path in tqdm( os.listdir(PRED_DATA_DIR) ):
        try:
            image_mask = cv2.imread(   os.path.join(MASK_DATA_DIR, image_path)   ) 
            ground_truth.extend(image_mask[image_mask!=255].flatten()) 
            
            image_pred = cv2.imread(   os.path.join(PRED_DATA_DIR, image_path)   ) 
            predictions.extend(image_pred[image_mask!=255].flatten()) 
            
            
            
            """
            image_ground = cv2.imread(   os.path.join(RAW_DATA_DIR, image_path)   ) 
            
            image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2GRAY) 
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY) 
            
            plt.imshow(image_ground)
            plt.show() 
            
            plt.imshow(image_pred, cmap = 'jet', vmin=0, vmax=10)
            plt.show() 
            
            
            plt.imshow(image_mask, cmap = 'jet', vmin=0, vmax=10)
            plt.show() 
            """
            
            
            
            
            
            
        except:
            pass 
        
    cm = confusion_matrix( ground_truth, predictions, normalize = 'true' )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap = 'plasma')
    disp.ax_.get_images()[0].set_clim(0, 1)
    fig = plt.gcf()
    fig.set_size_inches(10,10)
    plt.title("Confusion Matrix - Pixel Basis\nTrue Normalized")
    plt.xticks(rotation=90)
    plt.show() 
    
    cm = confusion_matrix( ground_truth, predictions, normalize = 'pred' )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap = 'plasma')
    disp.ax_.get_images()[0].set_clim(0, 1)
    fig = plt.gcf()
    fig.set_size_inches(10,10)
    plt.title("Confusion Matrix - Pixel Basis\nPredicted Normalized")
    plt.xticks(rotation=90)
    plt.show() 
    
    
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  