"""
16/03
Functions for data loading

# Notas del dataset : 
# 1 - Algunos datasets hacen el recorrido 0-360, aunque otros hacen 0-359. 
# Chequear condición para dado caso
# 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import re
from tqdm import tqdm
import SimpleITK as sitk
import pickle
x = 1

class ZebraDataset:
  '''
  Zebra dataset 
  Params:
    - folderPath (string): full path folder 
  '''
  def __init__(self, folderPath):

    self.folderPath = pathlib.Path(folderPath)
    self.folderName = pathlib.PurePath(self.folderPath).name

    self.objective = 10
    self.fileList = self._searchAllFiles(self.folderPath)

    if '4X' in self.folderName:

      self.objective = 4

    # For loaded dataset
    self.fishPart = {'head': 0,
                     'body': 1,
                     'upper tail': 2, 
                     'lower tail' : 3}
    # For file string
    self.fishPartCode = {'head': 's000',
                        'body': 's001',
                        'upper tail': 's002', 
                        'lower tail' : 's003'}                      

    self.dataset = {}
    self.registeredDataset = None

  def _searchAllFiles(self, x):

    dirpath = x
    assert(dirpath.is_dir())
    file_list = []
    
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(self._searchAllFiles(x))
    

    return file_list

  def loadImages(self, sample = None):
    '''
    Params:
      - sample (string): {None, body, head, tail}
    '''
    
    fishPart = self.fishPartCode[sample]
    loadList = [f for f in self.fileList if ('tif' in str(f))]

    if sample is not None:

      loadList = [f for f in loadList if (fishPart in str(f))]

    loadImages = []
    loadAngle = []
    loadSample = []
    loadChannel = []
    loadTime = []

    angle = re.compile('a\d+')
    sample = re.compile('s\d+')
    channel = re.compile('c\d+')
    time = re.compile('t\d+')
    
    for f in tqdm(loadList):
      
      loadImages.append(np.array(Image.open(f)))
      loadAngle.append(float(angle.findall(str(f))[0][1:]))
      loadSample.append(float(sample.findall(str(f))[0][1:]))
      loadChannel.append(float(channel.findall(str(f))[0][1:]))
      loadTime.append(float(time.findall(str(f))[0][1:]))

    self.dataset = pd.DataFrame({'Filename':loadList, 
                                 'Image':loadImages,
                                 'Angle':loadAngle,
                                 'Sample':loadSample,
                                 'Channel':loadChannel,
                                 'Time':loadTime})

    # Create Registered Dataset - empty till reggistering
    self.registeredDataset = pd.DataFrame(columns = ['Image', 'Angle', 'Sample'])

    # Sort dataset by sample and angle
    self.dataset = self.dataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)

  def registerDataset(self, sample, inPlace = False):

    """
    Registers full dataset, by sample.
    Params:
      -sample (string): fish part sample.
    """  
    
    # self. = self.dataset[df.dataset.Sample == '000'].sort_values('Angle', axis = 0).reset_index(drop=True)
    sample = self.fishPart[sample]
    dataset = self.dataset[self.dataset.Sample == sample].filter(['Angle','Image'])

    # Assert angle step {360, 720} -> (1, 0.5)
    if dataset['Angle'].max() in [359, 360]:
      self.maxAngle = 360
    else:
      self.maxAngle = 720
    # {0-179, 0-359}
    #pruebita paso max angle
    angles = np.arange(0, self.maxAngle//2, 1).astype(float)
    # Parámetros de transformación
    self.Tparams = pd.DataFrame(columns = ['theta', 'Tx', 'Ty', 'Sample', 'Angle'])

    # Filtrado Laplaciano mas grayscale
    rglaplacianfilter = sitk.LaplacianRecursiveGaussianImageFilter()
    rglaplacianfilter.SetSigma(6)
    rglaplacianfilter.SetNormalizeAcrossScale(True)

    grayscale_dilate_filter = sitk.GrayscaleDilateImageFilter()
    IsoData = sitk.IsoDataThresholdImageFilter()
    
    # Registration algorithm
    R = sitk.ImageRegistrationMethod()
    
    # Similarity metric, optimizer and interpolator
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,
                                              numberOfIterations=10)
    R.SetInterpolator(sitk.sitkLinear)
    
    # Connect all of the observers so that we can perform plotting during registration.
    # R.AddCommand(sitk.sitkStartEvent, start_plot)
    # R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))
    
    for angle in tqdm(angles):
      
      fixed =  dataset[dataset.Angle == angle].iloc[0]['Image'].astype(float)
      moving = np.flipud(dataset[dataset.Angle == angle+self.maxAngle//2].iloc[0]['Image'].astype(float))
      # pair of images sitk
      fixed_s = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat32)
      moving_s = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat32)

      # f stands for filtered image
      fixed_s_f  = rglaplacianfilter.Execute(fixed_s)
      fixed_s_f = grayscale_dilate_filter.Execute(fixed_s)
      fixed_s_f = sitk.BinaryFillhole(IsoData.Execute(fixed_s))

      moving_s_f  = rglaplacianfilter.Execute(moving_s)
      moving_s_f = grayscale_dilate_filter.Execute(moving_s)
      moving_s_f = sitk.BinaryFillhole(IsoData.Execute(moving_s))

      # Initial Transform - Aligns center of mass (same modality - no processing/filtering)
      initialT = sitk.CenteredTransformInitializer(fixed_s, 
                                        moving_s, 
                                        sitk.Euler2DTransform(), 
                                        sitk.CenteredTransformInitializerFilter.MOMENTS)
  
      R.SetInitialTransform(initialT)
      
      fixed_s_f = sitk.Cast(fixed_s_f, sitk.sitkFloat32)
      moving_s_f = sitk.Cast(moving_s_f, sitk.sitkFloat32)

      outTx = R.Execute(fixed_s_f, moving_s_f) # Rotation + traslation
      params = outTx.GetParameters()
      self.Tparams = self.Tparams.append({'theta':params[0],
                                          'Tx':params[1],
                                          'Ty':params[2],
                                          'Sample':sample,
                                          'Angle':angle}, ignore_index=True)      

      # Check rotation

      # If inPlace, registration is applied to all the dataset, translating images
      # to half the value of vertical translation
      if inPlace == True:
        
        F2C_T = sitk.TranslationTransform(2)
        M2C_T = sitk.TranslationTransform(2)

        F2C_T.SetParameters((0, -params[2]/2))  # Fixed image to center
        M2C_T.SetParameters((0, params[2]/2))  # Moving image to center

        fixed_s_T = sitk.Resample(fixed_s, 
                                  F2C_T, 
                                  sitk.sitkLinear, 
                                  0.0,
                                  fixed_s.GetPixelID())

        moving_s_T = sitk.Resample(moving_s, 
                                  M2C_T, 
                                  sitk.sitkLinear, 
                                  0.0,
                                  moving_s.GetPixelID())
        # Append to registered dataset
        self.registeredDataset = self.registeredDataset.append({'Image' : sitk.GetArrayFromImage(fixed_s_T),
                                      'Angle': angle,
                                      'Sample': sample}, ignore_index=True)
        self.registeredDataset = self.registeredDataset.append({'Image' : np.flipud(sitk.GetArrayFromImage(moving_s_T)),
                                      'Angle': angle+self.maxAngle,
                                      'Sample': sample}, ignore_index=True)
    
    # Order by angle
    self.registeredDataset = self.registeredDataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)
  
  def applyRegistration(self, sample):
    """
    Applies registration for dataset from registration params
    """

    assert(self.Tparams is not None)
    
    sample_code = self.fishPart[sample]
    dataset = self.dataset[self.dataset.Sample == sample_code].filter(['Angle','Image'])

    # Assert angle step {360, 720} -> (1, 0.5)
    if dataset['Angle'].max() in [359, 360]:
      self.maxAngle = 360
    else:
      self.maxAngle = 720

    angles = np.arange(0, self.maxAngle//2, 1).astype(float)

    self.registeredDataset = pd.DataFrame(columns = ['Image', 'Angle', 'Sample'])

    for angle in tqdm(angles):

      fixed =  dataset[dataset.Angle == angle].iloc[0]['Image'].astype(float)
      moving = dataset[dataset.Angle == angle+self.maxAngle//2].iloc[0]['Image'].astype(float)

      fixed_s = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat32)
      moving_s = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat32)

      params = self.Tparams[self.Tparams.Angle == angle].iloc[0]['Ty']
      
      # setting moving transform
      transform = sitk.TranslationTransform(2)
      transform.SetParameters((0, -params/2))  # Fixed image to center
      
      fixed_s_T = sitk.Resample(fixed_s, 
                                    transform, 
                                    sitk.sitkLinear, 
                                    0.0,
                                    fixed_s.GetPixelID())

      moving_s_T = sitk.Resample(moving_s, 
                                    transform, 
                                    sitk.sitkLinear, 
                                    0.0,
                                    moving_s.GetPixelID())
      # Append to registered dataset
      self.registeredDataset = self.registeredDataset.append({'Image' : sitk.GetArrayFromImage(fixed_s_T),
                                        'Angle': angle,
                                        'Sample': sample}, ignore_index=True)
      self.registeredDataset = self.registeredDataset.append({'Image' : sitk.GetArrayFromImage(moving_s_T),
                                        'Angle': angle+self.maxAngle//2,
                                        'Sample': sample}, ignore_index=True)
    
    self.registeredDataset = self.registeredDataset.sort_values(['Sample','Angle'], axis = 0).reset_index(drop=True)
  
  def getRegisteredVolume(self, margin = 10, useSegmented = False):
    '''
    Returns registered and stacked numpy volume, ordered by angle
    Calculates lower and upper non-zero limits for sinograms, with a safety
    margin given by margin.
    '''
    assert(self.registeredDataset is not None)

    self.registeredVolume = np.stack(self.registeredDataset['Image'].to_numpy())
    # Calculates non-zero boundary limit for segmenting the volume
    self.upperLimit = np.argmin(self.registeredVolume.sum(axis = (0,2)))-margin
    self.lowerLimit = self.registeredVolume.shape[1] - self.upperLimit
    
    # Normalize volume
    if useSegmented == True:
    
      return self.registeredVolume[:, self.lowerLimit:self.upperLimit, :]
    
    else:
    
      return self.registeredVolume
  
  def saveRegTransforms(self):
    
    with open(str(self.folderPath)+'transform.pickle', 'wb') as h:
      pickle.dump(self.Tparams,  h)

  def loadRegTransforms(self):

    with open(str(self.folderPath)+'transform.pickle', 'rb') as h:
      self.Tparams = pickle.load(h)

  def saveRegisteredDataset(self, pickleName = ''):

    with open(str(self.folderPath)+pickleName+'.pickle', 'wb') as pickleFile:
    
      pickle.dump({'reg_dataset' : self.registeredDataset,
                  'reg_transform' : self.Tparams}, pickleFile)

  def loadRegisteredDataset(self):

    with open(str(self.folderPath)+pickleName+'.pickle', 'rb') as pickleFile:
      
      reg = pickle.load(pickleFile)
      self.registeredDataset = reg['reg_dataset']
      self.Tparams = reg['reg_transform']

def subsample(volume, max_angle, angle_step, subsampling_type = 'linear'):
  '''
  Subsamples projections according to maximum angle. 
  Params : 
    volume (np.ndarray) : Projection volume, [angles, x-axis, z-axis]
    angle_step : For linear subsampling, reconstruction
    subsampling_type (string) :  Type of subsampling (linear, golden angle)
  
  Returns subsampled volume and angles.
  '''

  if subsampling_type == 'linear':

    beams = int(max_angle/angle_step)
    angles = np.linspace(0., max_angle-max_angle/beams, beams)
    
    return angles, volume[angles.astype(int),:,:]
  
  else:

    return angles, volume

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    # plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    print('Metric :', registration_method.GetMetricValue())
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask




