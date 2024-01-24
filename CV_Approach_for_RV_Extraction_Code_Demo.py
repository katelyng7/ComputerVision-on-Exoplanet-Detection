# A Computer Vision Approach to Radial Velocity Extraction for Exoplanet Detection 
# Demo Code
# Copyright 2023-2024 Katelyn Gan (katelyngan77@gmail.com).
# This program is free software; you can redistribute it and/or modfy it
# under the terms of the GNU Gneral Public License as published by 
# the Free Software Foundation.

import numpy as np
import cv2
#matplotlib.use('qtAgg') # do not display plot unless plt.show()
#matplotlib.use('Agg') # do not display plot unless plt.show()
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 100})
SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
from scipy import interpolate
from astropy.io import fits
import os
import pickle
from scipy.constants import c
import random
from scipy.signal import find_peaks
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
from astropy.timeseries import LombScargle
import scipy.signal as signal
# import matplotlib.dates as mdates
# from matplotlib.ticker import AutoMinorLocator

# *************** FUNCTIONS ***************************************************************************************************************
def gen_chunks(DataSet, Avg_Spectrum_OverSampling, n_start, n_stop, cutoffRatio, widthMin, widthMax, heightminRatio):
    # function to pick the top chunks on orders from n_start to n_stop. 
    # use peakRatio to determin the start_pixel and stop_pixel for each chunk
    
    # open average spectrum
    with open(DataSet+'templates_ao'+str(Avg_Spectrum_OverSampling)+'_n'+str(n_start)+'_n'+str(n_stop)+'.pickle','rb') as f:
        totTemplate, wavelength, shift_per_pixel=pickle.load(f)
    chunkIndices = []
    start_pixel = []
    stop_pixel = []
    imgFolder = DataSet+'ChunkSelectionImg//v'+Chunk_Algorithm_Version+'_ao'+str(Avg_Spectrum_OverSampling)+'_nMin'+str(n_start)+'_nMax'+str(n_stop)+'_cutoff'+str(cutoffRatio)+'_widthMax'+str(widthMax)+'_heightMin'+str(heightminRatio)+'//'
    make_directories(imgFolder)
    for n in range(0, n_stop-n_start+1):
        print("Working on Order #{} ".format(n+n_start))
        x,y,z = find_local_extremum(totTemplate[n,:], 0, cutoffRatio, widthMin, widthMax, heightminRatio, n, imgFolder, Avg_Spectrum_OverSampling,wavelength)
        chunkIndices.append(x)
        start_pixel.append(y)
        stop_pixel.append(z)    
    # save with the avg template into a new file
    AvgSpectrum = 'AvgSpectrum_with_Chunks_ao'+str(Avg_Spectrum_OverSampling)+'_nMin'+str(n_start)+'_nMax'+str(n_stop)+'_cutoff'+str(cutoffRatio)+'_widthMax'+str(widthMax)+'_heightMin'+str(heightminRatio)+'_'+Chunk_Algorithm_Version
    savetofile = DataSet + AvgSpectrum + '.pickle'
    with open(savetofile,'wb') as f:
        pickle.dump([totTemplate, wavelength, shift_per_pixel, chunkIndices, start_pixel, stop_pixel], f)
    return

def find_local_extremum(input_array, max_or_min, cutoffRatio, widthMin, widthMax, heightminRatio, n,imgFolder, Avg_Spectrum_OverSampling,wavelength):
    if max_or_min:
        array = input_array
    else:
        array = -input_array[0::]+max(input_array)
    
    goodPeaks = []
    start_pixel = []
    stop_pixel = []
    
    # Find indices of local peaks    
    heightmin = max(array)*heightminRatio # set minimum peak height
    stop = False
    margin_pixel=50
    while not stop:        
        if Chunk_Algorithm_Version == '1.1':
            peaks, _ = find_peaks(array[margin_pixel:-margin_pixel], height=heightmin , distance=200, width=[1,200]) # find the wavelength lines
        else:
            peaks, _ = find_peaks(array[margin_pixel:-margin_pixel], height=heightmin , distance=25*Avg_Spectrum_OverSampling, width=[1,25*Avg_Spectrum_OverSampling]) # find the wavelength lines
            
        if len(peaks) > 2: # more than two peaks is found
            stop = True
        else:            
            heightmin = heightmin*0.8 # reduce hightmin            
        
    if not peaks.size == 0: # not empty
        peaks += margin_pixel
        
        # #Plot the spectra and candidate peaks
        # plt.figure(figsize=(8,4),dpi=200)
        # plt.plot(array)
        # plt.plot(peaks, array[peaks], 'x')
        # plt.show() 
        
        # Calculate start_pixel and stop_pixel for each peak
        for j, peak in enumerate(peaks):
            peak_value = array[peak]
    
            # Find left and right indices 
            for i in range(peak - 1, -1, -1):
                if array[i] < peak_value*cutoffRatio:
                    leftPixel = i
                    break                
                else:
                    leftPixel = 0
            for i in range(peak + 1, len(input_array)):
                if array[i] < peak_value*cutoffRatio:
                    rightPixel = i
                    break   
                else:
                    rightPixel = array.size-1
            if rightPixel-leftPixel <= widthMax and rightPixel-leftPixel >= widthMin:
                goodPeaks.append(peak)
                start_pixel.append(leftPixel)
                stop_pixel.append(rightPixel)      
        
    return goodPeaks, start_pixel, stop_pixel

def make_directories(path):
    try:
        os.makedirs(path)
        print(f"Directories were created successfully at path: \r\n{path}\r\n")
    except FileExistsError:
        print(f"Directories already exist at path: \r\n{path}\r\n")
    except Exception as e:
        print(f"An error occurred while creating directories: {str(e)}")
        
def make_order_directories(parent_directory):
    # Create new directories
    for i in range(1, 86):
        directory_name = str(i)
        directory_path = os.path.join(parent_directory, directory_name)
        os.makedirs(directory_path)
        
def genRawSpectra_byChunks(AvgSpectrum_pickle_file, fitsFilelist, n_start, n_stop, v_Pixel, w_oversampling, uncertainty, savetoFolder):    
    # Make directory to save images    
    if not os.path.exists(savetoFolder):
        make_directories(savetoFolder)
        make_order_directories(savetoFolder)
        
    # load template and chunk data
    with open(AvgSpectrum_pickle_file,'rb') as f:
        totTemplate, wavelength, shift_per_pixel, chunkIndices, start_pixel_array, stop_pixel_array = pickle.load(f)  
    wavelengthOption = 'bary_wavelength'
    
    
    for n in range(0, n_stop-n_start+1):
        N = n + n_start
        print("Working on Order #{}".format(N))
        # Create avgSpectrum images
        start_pixel_array_byorder = start_pixel_array[n]
        for i, start_pixel in enumerate(start_pixel_array_byorder): 
            stop_pixel = stop_pixel_array[n][i]
            # read the start and stop wavelength
            logwavelength_low = wavelength[n, start_pixel]
            logwavelength_high = wavelength[n, stop_pixel]
            # Perform oversampling  
            numWavelength = (stop_pixel-start_pixel)*w_oversampling+1
            logwavelength = np.linspace(logwavelength_low, logwavelength_high, numWavelength)             
            f = interpolate.interp1d(wavelength[n,:], totTemplate[n,:]) 
            spectrum_interp_img1 = f(logwavelength)
            # build file name to be saved
            ImgFileName1 = savetoFolder+'//'+str(N)+'//Raw_v' + str(v_Pixel) + '_w'+'_n' + str(N) + '_p' + str(start_pixel) + '_to_p' + str(stop_pixel) + '_OS' + str(w_oversampling)+'_AvgTemplate.jpg'
            if not os.path.exists(ImgFileName1):                          
                # Construct image
                spec_image_img1 = np.zeros((v_Pixel, numWavelength))   
                spec_image_img1[:] = spectrum_interp_img1
                normalized_image = (spec_image_img1 - np.min(spec_image_img1)) / (np.max(spec_image_img1) - np.min(spec_image_img1))
                grayscale_image = cv2.cvtColor((normalized_image*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                plt.imsave(ImgFileName1, grayscale_image)
        
        # Create individual spectrum images
        for fitsFile in fitsFilelist:            
            fitsFilePath = os.path.join(fitsFile_folder, fitsFile)
            with fits.open(fitsFilePath) as ff:
                data = ff[1].data
                ff.close()
            # Interpolate the test spectrum at the common wavelength    
            spectrum0 = data['spectrum'][N,:]/data['continuum'][N,:]        
            logwavelength0 = np.log(data[wavelengthOption][N,:])        
            if uncertainty:
                uncertainty0 =  data['uncertainty'][N,:]/data['continuum'][N,:] 
             
            for i, start_pixel in enumerate(start_pixel_array_byorder):
                stop_pixel = stop_pixel_array[n][i]
                # read the start and stop wavelength
                logwavelength_low = wavelength[n, start_pixel]
                logwavelength_high = wavelength[n, stop_pixel]
                # Perform oversampling  
                numWavelength = (stop_pixel-start_pixel)*w_oversampling+1
                logwavelength = np.linspace(logwavelength_low, logwavelength_high, numWavelength)
                f = interpolate.interp1d(logwavelength0, spectrum0) 
                spectrum_interp_img2 = f(logwavelength)                 
                ImgFileName2 = savetoFolder+'//'+str(N)+'//Raw_v' + str(v_Pixel) + '_w'+'_n' + str(N) + '_p' + str(start_pixel) + '_to_p' + str(stop_pixel) + '_OS' + str(w_oversampling)+ '_'+fitsFile[0:-5]+'.jpg'
                # Construct image
                spec_image_img2 = np.zeros((v_Pixel, numWavelength))     
                if uncertainty:             
                    f = interpolate.interp1d(logwavelength0, uncertainty0) 
                    uncertainty_interp_img2 = f(logwavelength)
                    # add uncertainty noise
                    for row in range(v_Pixel):
                        spec_image_img2[row,:] = spectrum_interp_img2 + np.random.randn(numWavelength)*uncertainty_interp_img2
                else:
                    # No noise. Set every row to spectrum_interp_img2
                    spec_image_img2[:] = spectrum_interp_img2
                            
                normalized_image = (spec_image_img2 - np.min(spec_image_img2)) / (np.max(spec_image_img2) - np.min(spec_image_img2))
                grayscale_image = cv2.cvtColor((normalized_image*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                plt.imsave(ImgFileName2, grayscale_image) 
    return 0

def find_multiples(numbers, limit):
    multiples = set()
    for number in numbers:
        for i in range(1, limit):
            if number * i < limit:
                multiples.add(number * i)
    return multiples

def mask(grayscale2DArray1, grayscale2DArray2, MaskType, LocationType, intensity_max, linegap):
    # Apply Mask to the input grayscale2DArray
    # Input: [MaskType, LocationType, intensity, linegap] 
    # - MaskTypes = [0, 1, 2] # 0: fixed; 1: gradient; 2: random 
    # - LocationTypes = [0,1] # 0: multiple of a single factor (evenly distributed); 1: multiples of several factors; 2: random
    
    # Find the multiples of numbers in linegap
    rowNum, columnNum = grayscale2DArray1.shape
    lineSet = find_multiples(linegap, rowNum)     
    lineCount = len(lineSet)   
    if LocationType==0: 
        lines = np.array(list(lineSet))
    else:            
        lines = np.array(random.sample(range(rowNum), lineCount))
        
    if MaskType==0: # fixed          
        intensity = np.ones(lineCount)*intensity_max  
    elif MaskType==1: # gradient
        intensity = np.linspace(0, intensity_max, lineCount).astype(np.uint16)
    else: # random
        intensity = np.random.randint(intensity_max+1, size=lineCount)
    
    lines.sort()
    for i, row in enumerate(lines):  
        grayscale2DArray1[row,:] = intensity[i]
        grayscale2DArray2[row,:] = intensity[i]
    return [grayscale2DArray1, grayscale2DArray2]

def features_matching(Image1, Image2, g_match_threshold, showImg):   
    # Function for Feature Matching + Perspective Transformation
    # Check input type - img file path or grayscale 2D array
    if type(Image1 ) == str:
        img1 = cv2.imread(Image1, 0)   # read train image in grayscale
    else:
        img1 = Image1
        
    if type(Image2 ) == str:
        img2 = cv2.imread(Image2, 0)   # read train image in grayscale
    else:
        img2 = Image2
        
    try:
        if showImg==2:        
            plt.figure(figsize=(18,10), dpi=100)
            plt.imshow(img1)
            plt.title('Avg Spectrum')
            plt.figure(figsize=(18,10), dpi=100)
            plt.imshow(img2)
            plt.title('Individual Spectrum')
    
        min_match=1
        
        # SIFT detector
        sift = cv2.SIFT_create()
    
        # extract the keypoints and descriptors with SIFT
    
        kps1, des1 = sift.detectAndCompute(img1,None)
        kps2, des2 = sift.detectAndCompute(img2,None)
        
        if showImg==2:
            # Display key points for reference image in green color
            imgWithKP = cv2.drawKeypoints(img1, kps1, 0, (0,255,0), None)
            imgWithKP1 = imgWithKP[:,:,0]
            imgWithKP = cv2.drawKeypoints(img2, kps2, 0, (0,255,0), None)
            imgWithKP2 = imgWithKP[:,:,0]
            imgshow = np.concatenate((imgWithKP1, imgWithKP2), axis=1)
            plt.figure(figsize=(18,10), dpi=100)
            plt.imshow(imgshow)
            plt.title('Spectrum w/ KeyPoint')
        
        featureCount = [len(kps1),len(kps2)]
        #print(featureCount)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
    
        flann = cv2.FlannBasedMatcher(index_params, search_params)
    
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[1,0] for i in range(len(matches))]    
        
    
        # store all the good matches (g_matches) as per Lowe's ratio 
        g_match_m = []
        g_match_mn = []
        for i, (m,n) in enumerate(matches):
            if m.distance < g_match_threshold* n.distance:
                g_match_mn.append([m,n])
                g_match_m.append(m)
                #matchesMask[i]=[1,0]
        
        num_g_match = len(g_match_m)
        #print('Good Match = {}'.format(num_g_match))
        # Draw all matches
        draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)
        img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, matches, None, **draw_params)
            
        # Draw good matches only
        matchesMask = [[1,0] for i in range(len(g_match_mn))]  
        draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)
        img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, g_match_mn, None, **draw_params)
            
        #cv2.imshow('Image Match', img3)
        
        matchesMask = [[1,0] for i in range(len(matches))]  
        
        if num_g_match>min_match:
            src_pts = np.float32([ kps1[m.queryIdx].pt for m in g_match_m ]).reshape(-1,1,2)
            dst_pts = np.float32([ kps2[m.trainIdx].pt for m in g_match_m ]).reshape(-1,1,2)
    
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
    
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
         
            draw_params = dict(matchColor = (0,255,255), singlePointColor = (0,255,0), matchesMask = matchesMask, flags = 2)  # only inliers   
            
            shift = (w-1)/2-np.average(dst[:,0,0])
            # delta_shift = np.sqrt(np.mean((pts[:,0,0]-dst[:,0,0])**2))
            
            # crop matching region and show image
            # matching_region = crop_region(path_train, c_p)
            if showImg:
                img3 = cv2.drawMatches(img1, kps1, img2, kps2, g_match_m, None, **draw_params)
                plt.figure(figsize=(18,10), dpi=100)
                plt.imshow(img3)
            # print("Image1 feature detected: {}; Image2 feature detected: {}. Good matches found: {}".format(featureCount[0], featureCount[1], len(g_match_m)))
            # return (shift, delta_shift, featureCount, num_g_match, M)    
            return (shift, featureCount, num_g_match, M)    
        else:
            # print("Image1 feature detected: {}; Image2 feature detected: {}. Not enough matches have been found! - {}/{}".format(featureCount[0], featureCount[1], len(g_match_m), min_match))
            matchesMask = None
            # return (9999, 9999, featureCount, num_g_match, 0)
            return (9999,  featureCount, num_g_match, 0)
    except:
        if featureCount:
            # return (9999, 9999, featureCount, 0, 0)
            return (9999, featureCount, 0, 0)
        else:
            # return (9999, 9999, [0,0], 0, 0)
            return (9999, [0,0], 0, 0)
            
            

    else:
        print("Not enough matches have been found! - %d/%d" % (len(g_match_m), min_match))
        matchesMask = None
        # return (None, None, 9999, 9999, featureCount, num_g_match)
        return (None, None, 9999, featureCount, num_g_match)

def calculateSingleRV(RVPickleFile, RV_cutoff, fitsNum):
    with open('AvgSpectrum_with_Chunks_' + Chunk_Algorithm_Version + '.pickle','rb') as f:
        totTemplate, wavelength, shift_per_pixel, chunkIndices, start_pixel_array, stop_pixel_array = pickle.load(f)  
    del totTemplate, wavelength, chunkIndices, start_pixel_array, stop_pixel_array
    
    with open(RVPickleFile,'rb') as f:
        # shift_stats, delta_shift_stats, featureCount_stats, num_g_match_stats = pickle.load(f)
        shift_stats, featureCount_stats, num_g_match_stats = pickle.load(f)
        
    # extract parameters from the filename
    strArray = RVPickleFile.split('_')
    w_oversampling = int(re.findall(r'\d+', strArray[2])[0])
    n_start = int(re.findall(r'\d+', strArray[4])[0])
    n_stop = int(re.findall(r'\d+', strArray[4])[1])
    orderCount = n_stop-n_start
    RV_avg = np.zeros(4)
    
    i = fitsNum
    RV_fitsfile = np.zeros([orderCount, 4])
    for j, n in enumerate(range(n_start, n_stop)):
        shift_array = np.array(shift_stats[i][j])
        RVs = (np.exp(shift_per_pixel[n]/w_oversampling*shift_array)-1)*c
        RVs_good = RVs[abs(RVs)<RV_cutoff]
        RV_fitsfile[j,0] = np.average(RVs)
        RV_fitsfile[j,1] = np.var(RVs)
        RV_fitsfile[j,2] = np.average(RVs_good)
        RV_fitsfile[j,3] = np.var(RVs_good)
    RV_avg[0] = np.average(RV_fitsfile[:,0]) # avg of all RVs
    RV_weights = 1/RV_fitsfile[:,1] # use variance as weights
    RV_avg[1] = np.average(RV_fitsfile[:,0], weights=RV_weights) # weighted avg of all RVs
    
    RV_avg[2] = np.average(RV_fitsfile[:,2]) # avg of good RVs
    RV_good_weights = 1/RV_fitsfile[:,3] # use variance of good RVs as weights
    RV_avg[3] = np.average(RV_fitsfile[:,2], weights=RV_good_weights) # weighted avg of good RVs
        
    return -RV_avg

def calculateRV(RVPickleFile_fullpath, RV_cutoff, w_oversampling, n_start, n_stop, shift_per_pixel):   
    with open(RVPickleFile_fullpath,'rb') as f:
        # shift_stats, delta_shift_stats, featureCount_stats, num_g_match_stats = pickle.load(f)
            shift_stats,  featureCount_stats, num_g_match_stats = pickle.load(f)
        
    orderCount = n_stop-n_start+1
    fitsCount = len(shift_stats)
    RV_avg = np.zeros([fitsCount, 4])
    
    for i in range(fitsCount):
        RV_fitsfile = np.zeros([orderCount, 4])
        for j, n in enumerate(range(n_start, n_stop+1)):
            # print('i={}, j={}'.format(i,j))
            shift_array = np.array(shift_stats[i][j])
            RVs = (np.exp(shift_per_pixel[j]/w_oversampling*shift_array)-1)*c
            RVs_good = RVs[abs(RVs)<RV_cutoff]
            RV_fitsfile[j,0] = np.average(RVs)
            RV_fitsfile[j,1] = np.var(RVs)
            RV_fitsfile[j,2] = np.average(RVs_good)
            RV_fitsfile[j,3] = np.var(RVs_good)
        RV_avg[i, 0] = np.average(RV_fitsfile[:,0]) # avg of all RVs
        RV_weights = 1/RV_fitsfile[:,1] # use variance as weights
        RV_avg[i, 1] = np.average(RV_fitsfile[:,0], weights=RV_weights) # weighted avg of all RVs
        
        RV_avg[i, 2] = np.average(RV_fitsfile[:,2]) # avg of good RVs
        RV_good_weights = 1/RV_fitsfile[:,3] # use variance of good RVs as weights
        RV_avg[i, 3] = np.average(RV_fitsfile[:,2], weights=RV_good_weights) # weighted avg of good RVs
        
    return -RV_avg

def check_string_in_nested_list(nested_list, search_string):
    for sublist in nested_list:
        if isinstance(sublist, list):
            if check_string_in_nested_list(sublist, search_string):
                return True
        elif isinstance(sublist, str) and search_string in sublist:
            return True
    return False

# *************** END OF FUNCTIONS ************************************************************************************************************

## *************** Code Settings ***************************************************************************************************************
# General Settings
EXPRES_folder = 'EXPRES//'                                              # EXPRES Data Folder
DataSavingRootFolder = 'RV//'                                           # Root folder to save the result data
Star = '101501'                                                         # Define which star to analyze ('101501', '26965', '10700', '34411', etc.)
# Star = '34411'  
# Star = '26965'  
# Star = '10700'

# Step 1: Average Spectrum Calculation Settings
startInd = 1000                                                         # Starting index of spectrum calculation
stopInd = 7001                                                          # Stop index of spectrum calculation
Avg_Spectrum_OverSamplings = [4]                                        # List of Target Average Spectrum Over Samplings. Example: [1, 2, 4, 8]

# Step 2: Absorption Lines/Chunks Selection Settings
n_pairs = [[40, 60]]
Chunk_Algorithm_Version = '2.4'                                         # Algorithm Version #
cutoffRatio = 0.75                                                      # find the start and stop pixels at which the spectrum height is cutoffRatio*chunk height
heightminRatio = 0.3                                                # Minimum chunk height = heightminRatio * maximum spectrum height

# Step 3: Raw Spectrum Image Generation Settings                                                   
v_Pixels = [100]                                                        # List of Vertical Pixels. Example: [100, 200]
w_oversamplings = [2]                                                   # List of Wavelength Over Samplings. Example: [1, 4, 8]
uncertainty = False                                                     # Use uncertainty as noise

# Step 4: OpenCV feature matching and shift calculation Settings                                # List of [n_start_OpenCV, n_stop_OpenCV] pairs. [n_start_OpenCV, n_stop_OpenCV] is the range of orders to use in this step. n_start_OpenCV > n_start; n_stop_OpenCV < n_stop
intensities = [200]                                                     # grayscale intensity for the horizontal lines
linegap = [7, 11, 23, 37, 47]                                      # location factors for the horizontal lines. 
MaskTypes = [0]                                                            # Mask type: 0: fixed; 1: gradient; 2: random 
LocationType = 0                                                        # Horizontal lines location: 0: multiples of several factors; 1: random
discretization_ratios = [0.05]                                           # discretization settings: the smaller the hearier
showImg = 0                                                             # Image plotting: 0: no plotting; 1: Plot matching only; 2: Plot all template, test, and matching images
g_match_thresholds = [1]                                                # Feature matching threshold. Matching distance lower than the number will be considered good match and be used for homography matrix calculation 
useContrastEnhancement = True                                           # Use contrast enhancement or not
   
## ************** END OF Code Settings *********************************************************************************************************



# ## ************* Code initialization. Do Not Modify *****************************************************************************************************************
# Build AvgSpectrum keywords
DataSet = Star+'//'                                                                 # Star's sub-directory
fitsFile_folder = EXPRES_folder + DataSet + "Spectra " + Star + "//"
fitsFiles = [f for f in os.listdir(fitsFile_folder) if f.endswith('.fits')]         # Get a list of all the FITS files in the folder 
# fitsFiles = fitsFiles[0:10]
fitsCount = len(fitsFiles)
RawSpectrumImg_folder = DataSet+'RawSpectrumImg//'
RV_Results_folder = DataSavingRootFolder + DataSet
folders = ['ChunkSelectionImg', 'RawSpectrumImg'] # Setup Directories
print("\r\n**********************************************************************************************************************")
print("\r\n                    A Computer Vision Approach to Radial Velocity Extraction for Exoplanet Detection                  ")
print("                                                     Demo Code                                                            ")
print("                               Copyright 2023-2024 Katelyn Gan (katelyngan77@gmail.com)                                   ")
print("\r\n**********************************************************************************************************************")


print("\r\nSet up directories:")
make_directories(Star)
make_directories(DataSavingRootFolder)
for folder in folders:
    make_directories((Star+'//' + folder))      
# # ************************************************************************************************************************************************************

# Record the start time
start_time = time.time()

for n_start, n_stop in n_pairs: # loop on n_pairs
    AvgSpectrums = []
    for Avg_Spectrum_OverSampling in Avg_Spectrum_OverSamplings:
        widthMin = 2*Avg_Spectrum_OverSampling
        widthMax = 12*Avg_Spectrum_OverSampling
        AvgSpectrums.append('AvgSpectrum_with_Chunks_ao'+str(Avg_Spectrum_OverSampling)+'_nMin'+str(n_start)+'_nMax'+str(n_stop)+'_cutoff'+str(cutoffRatio)+'_widthMax'+str(widthMax)+'_heightMin'+str(heightminRatio)+'_'+Chunk_Algorithm_Version)
    EXPRES_Num_Orders=n_stop - n_start + 1  
    
    # Step 1: Calculate the average spectrum and save data to a pickle file
    print('\r\n**************** Step 1: Calculate the average spectrum and save data to a pickle file *******************************')
    
    for Avg_Spectrum_OverSampling in Avg_Spectrum_OverSamplings:
        numWavelength = (stopInd-startInd-1)*Avg_Spectrum_OverSampling+1
        totTemplate = np.zeros((EXPRES_Num_Orders,numWavelength))
        LogWavelengths = np.zeros((EXPRES_Num_Orders,numWavelength))
        shift_per_pixel = np.zeros(EXPRES_Num_Orders)   
        
        # Loop over each FITS file, read its data, and add to template wavelength, produces template 
        for i in range(EXPRES_Num_Orders):
            order = i+n_start
            template=np.zeros(numWavelength)
            print('Working on Order #{}'.format(order))
            for j, file_name in enumerate(fitsFiles):       
                file_path = os.path.join(fitsFile_folder, file_name)
                hdulist = fits.open(file_path)
                data = hdulist[1].data
                hdulist.close()
                
                if j==0:
                    logWavelength = np.log(data['bary_wavelength'][order][startInd:stopInd])
                    normalizedData = data['spectrum'][order][startInd:stopInd]/data['continuum'][order][startInd:stopInd]
                    startLogWavelength = logWavelength[0]
                    stopLogWavelength = logWavelength[-1]
                    LogWavelengths[i] = np.linspace(startLogWavelength, stopLogWavelength, numWavelength)
                    shift_per_pixel[i] = (stopLogWavelength-startLogWavelength)/numWavelength
                else:
                    logWavelength = np.log(data['bary_wavelength'][order])
                    normalizedData = data['spectrum'][order]/data['continuum'][order]
                    
                f = interpolate.interp1d(logWavelength, normalizedData)
                template = template + f(LogWavelengths[i])
                
            totTemplate[i]=template
            
        filename = DataSet+'templates_ao'+str(Avg_Spectrum_OverSampling)+'_n'+str(n_start)+'_n'+str(n_stop)+'.pickle'
        with open(filename, 'wb') as f:
            pickle.dump([totTemplate, LogWavelengths , shift_per_pixel], f)
    
    # Step 2: Find all good absorption lines (i.e. chunks) in each order of the average spectra and save the data along with the average spectrum to a new pickle file
    print('\r\n************* Step 2: Find all good absorption lines of the average spectra and save the data  *********************')
    for Avg_Spectrum_OverSampling in Avg_Spectrum_OverSamplings:
        widthMin = 2*Avg_Spectrum_OverSampling
        widthMax = 12*Avg_Spectrum_OverSampling # Maximum chunk width to consider. Ignore all chunks that has a width > widthMax
        gen_chunks(DataSet, Avg_Spectrum_OverSampling, n_start, n_stop, cutoffRatio, widthMin, widthMax, heightminRatio)
    
    # Step 3. Generate raw average spectrum and individual fitsfile spectrum images for good lines
    print('\r\n************** Step 3: Generate raw average spectrum and individual spectrum images for good lines *****************')     
    for Avg_Spectrum_OverSampling in Avg_Spectrum_OverSamplings:
        widthMin = 2*Avg_Spectrum_OverSampling
        widthMax = 12*Avg_Spectrum_OverSampling
        AvgSpectrum ='AvgSpectrum_with_Chunks_ao'+str(Avg_Spectrum_OverSampling)+'_nMin'+str(n_start)+'_nMax'+str(n_stop)+'_cutoff'+str(cutoffRatio)+'_widthMax'+str(widthMax)+'_heightMin'+str(heightminRatio)+'_'+Chunk_Algorithm_Version
        for v_Pixel in v_Pixels:
            AvgSpectrum_pickle_file = DataSet + AvgSpectrum +'.pickle'
            for w_oversampling in w_oversamplings:
                print('AvgSpectrum = {}, w_oversampling = {}'.format(AvgSpectrum, w_oversampling))
                savetoFolder = RawSpectrumImg_folder + AvgSpectrum +'_v' + str(v_Pixel) + '_w'+'_OS' + str(w_oversampling)+'_uncrty'+str(uncertainty)+'_grayscale//'
                genRawSpectra_byChunks(AvgSpectrum_pickle_file, fitsFiles, n_start, n_stop, v_Pixel, w_oversampling, uncertainty, savetoFolder)
                
    # Step 4. Calculate shift on all good chunks and save the data to a pickle file
    print('\r\n************* Step 4: Apply enhancement, calculate shift on all good lines, and save the data  *********************')
    for MaskType in MaskTypes:
        for Avg_Spectrum_OverSampling in Avg_Spectrum_OverSamplings:
            widthMin = 2*Avg_Spectrum_OverSampling
            widthMax = 12*Avg_Spectrum_OverSampling
            AvgSpectrum ='AvgSpectrum_with_Chunks_ao'+str(Avg_Spectrum_OverSampling)+'_nMin'+str(n_start)+'_nMax'+str(n_stop)+'_cutoff'+str(cutoffRatio)+'_widthMax'+str(widthMax)+'_heightMin'+str(heightminRatio)+'_'+Chunk_Algorithm_Version
            
            for v_Pixel in v_Pixels:   
                with open(DataSet + AvgSpectrum +'.pickle','rb') as f:
                    totTemplate, wavelength, shift_per_pixel, chunkIndices, start_pixel_array, stop_pixel_array = pickle.load(f)        
                for intensity in intensities:
                    for discretization_ratio in discretization_ratios:
                        for w_oversampling in w_oversamplings:
                            for g_match_threshold in g_match_thresholds:
                                print('n_start={}, n_stop={}, chunk_file={}, WO={}, g_threshold={}'.format(n_start, n_stop, AvgSpectrum, w_oversampling, g_match_threshold))
                                RV_save_folder =DataSavingRootFolder + DataSet + AvgSpectrum
                                make_directories(RV_save_folder)    
                                
                                pickleFileName = RV_save_folder+'//'+'v' + str(v_Pixel) + '_w'+'_OS' + str(w_oversampling)+'_uncrty'+str(uncertainty)+'_n'+str(n_start)+'-'+str(
                                        n_stop)+'_LnGap'+str(linegap)+'_gThrhld'+str(g_match_threshold)+'_MTyp'+str(MaskType)+'_Loc'+str(
                                            LocationType)+ '_discrt'+str(discretization_ratio)+'_int'+str(intensity)+'_fitsCnt'+str(fitsCount)+'_CE'+str(useContrastEnhancement)+'.pickle'
                                
                                shift_stats = []
                                # delta_shift_stats = []
                                featureCount_stats = []
                                num_g_match_stats = []
                                
                                RawSpectrumImg_folder_work = RawSpectrumImg_folder + AvgSpectrum + '_v' + str(v_Pixel) + '_w'+'_OS' + str(w_oversampling)+'_uncrty'+str(uncertainty)+'_grayscale//'
                                for j, fitsFile in enumerate(fitsFiles):
                                    print('Working on fits file #{} - {}...'.format(j, fitsFile))
                                    shift_byFitFiles = []
                                    # delta_shift_byFitFiles = []
                                    featureCount_byFitFiles = []
                                    num_g_match_byFitFiles = []
                                    for n in range(0, n_stop-n_start+1):   
                                        N=n+n_start
                                        shift_byFitFiles_n = []
                                        # delta_shift_byFitFiles_n = []
                                        featureCount_byFitFiles_n = []
                                        num_g_match_byFitFiles_n = []
                                        start_pixel_array_byorder = start_pixel_array[n]
                                        for i, start_pixel in enumerate(start_pixel_array_byorder):
                                            stop_pixel = stop_pixel_array[n][i]
                                            rawSpectrumTemplateFile = RawSpectrumImg_folder_work+str(N)+'//Raw_v' + str(v_Pixel) + '_w'+'_n' + str(N) + '_p' + str(start_pixel) + '_to_p' + str(stop_pixel) + '_OS' + str(w_oversampling)+'_AvgTemplate.jpg'
                                            rawSpectrumTestImgFile  = RawSpectrumImg_folder_work+str(N)+'//Raw_v' + str(v_Pixel) + '_w'+'_n' + str(N) + '_p' + str(start_pixel) + '_to_p' + str(stop_pixel) + '_OS' + str(w_oversampling)+ '_'+fitsFile[0:-5]+'.jpg'                
                                        
                                            rawSpectrumTemplate = cv2.imread(rawSpectrumTemplateFile, cv2.IMREAD_GRAYSCALE) 
                                            rawSpectrumTestImg = cv2.imread(rawSpectrumTestImgFile, cv2.IMREAD_GRAYSCALE)                         
                                            
                                            # add mask to images
                                            rawSpectrumTemplate_masked, rawSpectrumTestImg_masked =  mask(rawSpectrumTemplate, rawSpectrumTestImg, MaskType, LocationType, intensity, linegap)
                                            
                                            rawSpectrumTemplate_masked_discretized = (rawSpectrumTemplate_masked*discretization_ratio).astype(np.uint8)
                                            # convert it to full grayscale
                                            rawSpectrumTemplate_masked_discretized = ((rawSpectrumTemplate_masked_discretized - rawSpectrumTemplate_masked_discretized.min())/(rawSpectrumTemplate_masked_discretized.max() - rawSpectrumTemplate_masked_discretized.min())*255).astype(np.uint8)
                                            
                                            rawSpectrumTestImg_masked_discretized = (rawSpectrumTestImg_masked*discretization_ratio).astype(np.uint8)
                                            # convert it to full grayscale
                                            rawSpectrumTestImg_masked_discretized = ((rawSpectrumTestImg_masked_discretized - rawSpectrumTestImg_masked_discretized.min())/(rawSpectrumTestImg_masked_discretized.max() - rawSpectrumTestImg_masked_discretized.min())*255).astype(np.uint8)
                                            
                                            if useContrastEnhancement: # perform histogram equalization to increae contrast
                                                rawSpectrumTemplate_masked_discretized_equalized = cv2.equalizeHist(rawSpectrumTemplate_masked_discretized)
                                                rawSpectrumTestImg_masked_discretized_equalized = cv2.equalizeHist(rawSpectrumTestImg_masked_discretized)
                                            else: # No contrast enhancement
                                                rawSpectrumTemplate_masked_discretized_equalized = rawSpectrumTemplate_masked_discretized
                                                rawSpectrumTestImg_masked_discretized_equalized = rawSpectrumTestImg_masked_discretized
                                            
                                            # test feature detection
                                            MaskSettings = [MaskType, LocationType]
                                            # shift, delta_shift, featureCount, num_g_match, M= features_matching(rawSpectrumTemplate_masked_discretized, rawSpectrumTestImg_masked_discretized, g_match_threshold, showImg)    
                                            shift,  featureCount, num_g_match, M= features_matching(rawSpectrumTemplate_masked_discretized_equalized, rawSpectrumTestImg_masked_discretized_equalized, g_match_threshold, showImg)    
                                            RV = (np.exp(shift_per_pixel[n]/w_oversampling*shift)-1)*c
                                            # print('{}, n={},{}-{}, Mask={}: shift = {}, delta shift = {}, discretization = {}, feature count = {}, good match = {}, M={}, RV_mask = {}'.format(fitsFile, n, pixel_start, pixel_stop, MaskSettings, shift, delta_shift, discretization_ratio, featureCount, num_g_match, M, RV))                
                                            # print('{}, n={},{}-{}, Mask={}: shift = {}, delta shift = {}, discretization = {}, feature count = {}, good match = {}, RV_mask = {}'
                                            #        .format(fitsFile, n, start_pixel, stop_pixel, MaskSettings, shift, delta_shift, discretization_ratio, featureCount, num_g_match, RV))     
                                            shift_byFitFiles_n.append(shift)
                                            # delta_shift_byFitFiles_n.append(delta_shift)
                                            featureCount_byFitFiles_n.append(featureCount)
                                            num_g_match_byFitFiles_n.append(num_g_match)
                                          
                                        shift_byFitFiles.append(shift_byFitFiles_n)
                                        # delta_shift_byFitFiles.append(delta_shift_byFitFiles_n)
                                        featureCount_byFitFiles.append(featureCount_byFitFiles_n)
                                        num_g_match_byFitFiles.append(num_g_match_byFitFiles_n)
                                        
                                        
                                    shift_stats.append(shift_byFitFiles)
                                    # delta_shift_stats.append(delta_shift_byFitFiles)
                                    featureCount_stats.append(featureCount_byFitFiles)
                                    num_g_match_stats.append(num_g_match_byFitFiles)
                                            
                                with open(pickleFileName,'wb') as f:
                                    # pickle.dump([shift_stats, delta_shift_stats, featureCount_stats, num_g_match_stats], f)
                                    pickle.dump([shift_stats, featureCount_stats, num_g_match_stats], f)





# Step 5: Calculate and plot RVs for all previously saved pickle files in a folder
print('\r\n************* Step 5: Calculate and plot RVs for all previously saved pickle files in a folder *********************')
RV_cutoff = 5000
# Load, plot, and save the activity csv
data_file = EXPRES_folder + DataSet+Star+'_activity.csv'
X = pd.read_csv(data_file)

# get a list of all directories under a DataSet
RVPickle_folders = [os.path.join(RV_Results_folder, name)+'//' for name in os.listdir(RV_Results_folder) if os.path.isdir(os.path.join(RV_Results_folder, name))]
for RVPickle_folder in RVPickle_folders:    
    # extract avererage spectrum chunk info file location
    AvgSpectrum_Chunks_File_Name = RVPickle_folder.split('//')[2]
    # load the shift_per_pixel data
    AvgSpectrum_Chunks_File = DataSet + AvgSpectrum_Chunks_File_Name + '.pickle'
    with open(AvgSpectrum_Chunks_File,'rb') as f:
        totTemplate, wavelength, shift_per_pixel, chunkIndices, start_pixel_array, stop_pixel_array = pickle.load(f)  
    del totTemplate, wavelength, chunkIndices, start_pixel_array, stop_pixel_array
    
    make_directories(RVPickle_folder+'RV_Plots') # make a directory to save plots
    PickleFiles = [p for p in os.listdir(RVPickle_folder) if p.endswith('.pickle')]  # Get a list of all the pickle files in the folder
    
    # check if there is a saved RV_data file
    RV_summary_file = RVPickle_folder + 'RV_summary.pickle'
    if os.path.exists(RV_summary_file): # there is a saved RV_data file
        with open(RV_summary_file, 'rb') as f:
            RV_data = pickle.load(f)    
    else:
        RV_data = [] # No prreviously saved RV_data, start with an empty array
    
    for RVPickleFile in PickleFiles:
        if RVPickleFile != 'RV_summary.pickle':
            RVPickleFile_noext = RVPickleFile.split('.pickle')[0]
            # # check if this file already processed previously        
            # if not check_string_in_nested_list(RV_data, RVPickleFile_noext):
            print('Pickle File = {}'.format(RVPickleFile))    
            # extract w_oversampling, n_start, n_stop from file name
            strArray = RVPickleFile.split('_')
            w_oversampling = int(re.findall(r'\d+', strArray[2])[0])
            n_start = int(re.findall(r'\d+', strArray[4])[0])
            n_stop = int(re.findall(r'\d+', strArray[4])[1])
            
            avgRV = calculateRV(RVPickle_folder+RVPickleFile, RV_cutoff, w_oversampling, n_start, n_stop, shift_per_pixel)
            
            
            RV_info = RVPickleFile.split('.pickle')[0]
            if not np.isnan(avgRV[:,1]).any(): # all RV data are good
                RV_RMS = np.sqrt(np.mean(avgRV[:,1]**2))
                fig = plt.figure(figsize=(5,2.5), dpi=100)
                plt.plot(X['Time [MJD]'], avgRV[:,1], '.', color='blue')                    
                titletext = 'CV Method: HD'+Star+' RVs'
                plt.title(titletext,fontsize=16)
                plt.xlabel('Time [MJD]')
                plt.ylabel('RV [m/s]')
                yl_low = int(min(avgRV[:,1])*1.8)
                yl_high = int(max(avgRV[:,1])*1.8)
                plt.ylim([yl_low, yl_high])
                plt.tight_layout()
                savetofile=RVPickle_folder + 'RV_Plots//' + RV_info + '_RVRMS' + str(round(RV_RMS, 3)) + '_dot.png'
                fig.savefig(savetofile)
                
                fig = plt.figure(figsize=(5,2.5), dpi=100)
                plt.plot(X['Time [MJD]'], avgRV[:,1], color='blue')
                plt.title(titletext,fontsize=16)
                plt.xlabel('Time [MJD]')
                plt.ylabel('RV [m/s]')
                plt.ylim([yl_low, yl_high])
                plt.tight_layout()
                savetofile=RVPickle_folder + 'RV_Plots//' + RV_info + '_RVRMS' + str(round(RV_RMS, 3)) + '_line.png'
                fig.savefig(savetofile)
                
                print('    RV (RMS) = {}'.format(RV_RMS))
                RV_data.append([RV_info, RV_RMS])
                # plt.close('all')
            else:
                print('    RV (RMS) = NaN detected.')
                RV_RMS = 99999
                RV_data.append([RV_info, RV_RMS])
                    
    # save RV summary                
    with open(RV_summary_file,'wb') as f:
        pickle.dump(RV_data, f)


# Step 6. Plot industry pipeline, i.e. the CBC RV method
print('\r\n************** Step 6: Plot industry CBC method RVs ****************************************************************')
# Load, plot, and save the activity csv
data_file = EXPRES_folder + DataSet+Star+'_activity.csv'
X = pd.read_csv(data_file)
if Star == '26965':
    yli = [-13, 13]
elif Star == '10700':
    yli = [-9, 9]
elif Star == '101501':
    yli = [-13, 13]

fig = plt.figure(figsize=(5,2.5), dpi=100)
plt.plot(X['Time [MJD]'], X['CBC RV [m/s]'], '.', color='red')
titletext = 'Industry CBC: HD'+Star+' RVs'
plt.title(titletext,fontsize=16)
plt.xlabel('Time [MJD]')
plt.ylabel('RV [m/s]')
plt.ylim(yli)
plt.tight_layout()

savetofile=DataSet + 'Original_EXPRES_CBC_dot.png'
fig.savefig(savetofile)

fig = plt.figure(figsize=(5,2.5), dpi=100)
plt.plot(X['Time [MJD]'], X['CBC RV [m/s]'], color='red')
plt.title(titletext,fontsize=16)
plt.xlabel('Time [MJD]')
plt.ylabel('RV [m/s]')
plt.tight_layout()
plt.ylim(yli)
plt.tight_layout()
savetofile=DataSet + 'Original_EXPRES_CBC_line.png'
fig.savefig(savetofile)
# ****************************************************************************************


# Step 7. Plot and compare Periodogram of both CV and industry methods 
print('\r\n******************** Step 7: Plot and compare Periodogram of both CV and industry methods **************************')
101501
star = '101501'
# Plot EXPRES CBC pipeline RV periodogram
data_file = EXPRES_folder + DataSet+Star+'_activity.csv'
X = pd.read_csv(data_file)
x = X['Time [MJD]']
y = X['CBC RV [m/s]']
# # use astropy.timeseries's LombScargle function to draw the periodogram
# frequency, power = LombScargle(x, y).autopower()
# plt.plot(frequency, power)   

# Calculate OpenCV RV
#RVPickle_folder = 'RV//101501//AvgSpectrum_with_Chunks_ao8_nMin40_nMax60_cutoff0.75_widthMax96_heightMin0.3_2.4//'
AvgSpectrum_Chunks_File_Name = RVPickle_folder.split('//')[2]
w_oversampling = 2
n_start = n_pairs[0][0]
n_stop = n_pairs[0][1]
RV_cutoff = 5000
AvgSpectrum_Chunks_File = DataSet + AvgSpectrum_Chunks_File_Name + '.pickle'
with open(AvgSpectrum_Chunks_File,'rb') as f:
        totTemplate, wavelength, shift_per_pixel, chunkIndices, start_pixel_array, stop_pixel_array = pickle.load(f)  
del totTemplate, wavelength, chunkIndices, start_pixel_array, stop_pixel_array   

PickleFiles = [p for p in os.listdir(RVPickle_folder) if p.endswith('.pickle')]  # Get a list of all the pickle files in the folder
for RVPickleFile in PickleFiles:
    if RVPickleFile != 'RV_summary.pickle':
        strArray = RVPickleFile.split('_')
        w_oversampling = int(re.findall(r'\d+', strArray[2])[0])
        n_start = int(re.findall(r'\d+', strArray[4])[0])
        n_stop = int(re.findall(r'\d+', strArray[4])[1])
        avgRV = calculateRV(RVPickle_folder+RVPickleFile, RV_cutoff, w_oversampling, n_start, n_stop, shift_per_pixel)
        y1 = avgRV[:,1]
        
        # nout = 10000
        # w = np.linspace(0.0001, 0.2, nout)  # signal.lombscargle use angular frequency as default
        # f = w/(2*np.pi) # calculate frequency (i.e. 1/T)
        # pgram = signal.lombscargle(x, y, w, normalize=True)
        # pgram1 = signal.lombscargle(x, y1, w, normalize=True)
        
        frequency = np.linspace(0, 20, 100000)
        ls = LombScargle(x, y)
        ls1 = LombScargle(x, y1)
        power = ls.power(frequency)         
        power1= ls1.power(frequency) 
        probabilities = [0.07, 0.05, 0.005, 0.0005]
        FAP = ls.false_alarm_level(probabilities)  
        FAP1 = ls1.false_alarm_level(probabilities)  
        
        fontsize = 18
        fig, ax = plt.subplots(layout='constrained', figsize=(15, 7))
        ax.plot(frequency, power, color = 'red', linewidth=2, alpha=0.7, label='Industry CBC Method')
        ax.plot(frequency, power1, color = 'blue', linewidth=2, alpha=0.7, label='Computer Vision Method')
        ax.set_xlabel('Frequency [1/Day]', size = fontsize+5)
        ax.set_ylabel('Normalized Power', size = fontsize+5)
        ax_xticks = np.linspace(0.0, 0.10, 11)
        ax.set_xticks(ax_xticks)
        # ax.set_title(RVPickleFile)
        
        def one_over(x):
            """Vectorized 1/x, treating x==0 manually"""
            x = np.array(x, float)
            near_zero = np.isclose(x, 0)
            x[near_zero] = np.inf
            x[~near_zero] = 1 / x[~near_zero]
            return x        
        # the function "1/x" is its own inverse
        inverse = one_over
        
        secax = ax.secondary_xaxis('top', functions=(one_over, inverse))
        secax_xticks=[1000000, 100.0,50.0,33.3,25.0,20.0,16.7,14.3,12.5,11.1,10.0]
        secax_xlabels=secax_xticks.copy()
        secax_xlabels[0]=r"$\infty$"
        secax.set_xticks(secax_xticks)
        secax.set_xticklabels(secax_xlabels)
        secax.get_xticklabels()[0].set_fontsize(21)
        secax.set_xlabel('Period [Day]', size = fontsize+5)
        ax.set_xlim([-0.000001, 0.10])
        plt.show()
        plt.tight_layout()
        plt.legend(loc='upper left', fontsize = str(fontsize))
        
        # add FAP lines
        plt.axhline(y=FAP1[1], color = 'black', linestyle = '--')
        plt.text(0.088, FAP[1]+0.0045, 'FAP=5%', ha='left', va='bottom', size = fontsize)
        plt.axhline(y=FAP1[2], color = 'green', linestyle = '--')
        plt.text(0.088, FAP[2]+0.0045, 'FAP=0.5%', ha='left', va='bottom', size = fontsize)
        plt.axhline(y=FAP1[3], color = 'orange', linestyle = '--')
        plt.text(0.088, FAP[3]+0.0045, 'FAP=0.05%', ha='left', va='bottom', size = fontsize)
        savetofile=RVPickle_folder + 'RV_Plots//' + RVPickleFile+'_Periodogram.png'
        fig.savefig(savetofile)
        # plt.close(fig)


# Record the stop time
stop_time = time.time()
# Calculate the elapsed time
elapsed_time = stop_time - start_time
# Print the elapsed time
print(f"\r\nTask Completed! Elapsed time: {elapsed_time:.6f} seconds")



    