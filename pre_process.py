
# coding: utf-8
'''
See article.
https://becominghuman.ai/transfer-learning-retraining-inception-v3-for-custom-image-classification-2820f653c557

@INPUT:
    1. Folder that contains images of items. The images are in PNG format. 
    The folder is train_img. It has 3215 images
    2. The file train.csv is a table that relates each file name to the item 
    name/label whose image is contained in the file.
@PROCESSING:
    1. Convert PNG to JPG images.
    2. Store the JPG images in subfolders according to the item names. Each subfolder thus
    will have images that belong to a single class of items. No two 

'''

import os  #,sys
#import h5py
import pandas as pd
#import numpy as np
from keras.preprocessing.image import ImageDataGenerator,\
                                        load_img, \
                                        img_to_array
#                                        array_to_img, \
#from keras.models import Sequential
#from keras.layers import Dropout, Flatten, Dense
#from keras import applications
import matplotlib.pyplot as plt
import seaborn as sns
import math
#get_ipython().run_line_magic('matplotlib', 'inline') #for running in notebook
from tqdm import tqdm
from PIL import Image


# ## Let's discover the different labels 

# In[4]:


data_root='.'
tbl_train_filename_itemname=pd.read_csv('train.csv')
#tbl_test_filename_itemname=pd.read_csv('test.csv')
#Where are the Contents of test.csv used?
print("There are ", \
      tbl_train_filename_itemname.label.nunique(),'labels')
itemLabel_itemCount_pairs = tbl_train_filename_itemname.label.value_counts()
[print("Item Label: {} \t\t Inventory Count: {}".format(label, count)) \
                 for label, count in itemLabel_itemCount_pairs.iteritems()]


# ## Let's see the distribution of each class in the dataset

# In[5]:


plt.figure(figsize = (12,6))
sns.barplot(itemLabel_itemCount_pairs.index, \
            itemLabel_itemCount_pairs.values, alpha = 0.9)
plt.xticks(rotation = 'vertical')
plt.xlabel('Image Labels', fontsize =12)
plt.ylabel('Counts', fontsize = 12)
plt.show()


# ## Put each training image into a sub folder corresponding to its label after converting to JPG format

# In[6]:
'''
ipdb> tbl_train_filename_itemname[:5]
   image_id    label
0  train_1a     rice
1  train_1b    candy
2  train_1c      jam
3  train_1d   coffee
4  train_2a  vinegar

ipdb> tbl_train_filename_itemname.values[0:5]
array([['train_1a', 'rice'],
       ['train_1b', 'candy'],
       ['train_1c', 'jam'],
       ['train_1d', 'coffee'],
       ['train_2a', 'vinegar']], dtype=object)

ipdb> img
array(['train_1a', 'rice'], dtype=object)

The folders train_img contains only .png images.

The file train.csv is a table that relates each file name
to the item name/label whose image is contained in the file. Presumably this 
table has been manually created and verified

'''
'''
Output from previous runs should be removed. This is to avaoid multiple
copies of the same file.
'''
import shutil
tmpPath = os.path.join(data_root,\
                             'train_initial_sets_of_images')
if os.path.exists(tmpPath):
    shutil.rmtree(tmpPath)
assert(not os.path.exists(tmpPath))

tmpPath = os.path.join(data_root,\
                             'train_augmented_sets_of_images')
if os.path.exists(tmpPath):
    shutil.rmtree(tmpPath)
assert(not os.path.exists(tmpPath))


print("Converting original png images to jpg images ...")
for a_row_of_the_tbl in tqdm(tbl_train_filename_itemname.values):
#for a_row_of_the_tbl in (tbl_train_filename_itemname.values):
    filename = a_row_of_the_tbl[0]
    itemname=a_row_of_the_tbl[1]
    image_file_orig = os.path.join(data_root,\
                                   'train_img',\
                                   filename + \
                                   '.png')
    subdir_for_images_with_same_itemname = \
                os.path.join(data_root,\
                             'train_initial_sets_of_images',\
                             itemname)
    image_file_target = \
                os.path.join(subdir_for_images_with_same_itemname, \
                             filename +\
                             '.jpg')
    #https://stackoverflow.com/questions/11064786/get-pixels-rgb-using-pil
    im=Image.open(image_file_orig)
    rgb_im=im.convert('RGB')
    if not os.path.exists(subdir_for_images_with_same_itemname):
        os.makedirs(subdir_for_images_with_same_itemname)
    rgb_im.save(image_file_target)  
    if not os.path.exists(os.path.join(data_root,\
                                       'train_augmented_sets_of_images',\
                                       itemname)):
        os.makedirs(os.path.join(data_root,\
                                 'train_augmented_sets_of_images',\
                                 itemname))
    rgb_im.save(os.path.join(data_root,\
                             'train_augmented_sets_of_images',\
                             itemname, \
                             filename + \
                             '.jpg'))


# ## Some agile data augmentation (to prevent overfitting) + class balance

# In[7]:


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

AUGMENTED_ITEM_COUNT=600
prefix_num = 0
dir_path_initial_sets_of_images = \
            os.path.join(data_root,'train_initial_sets_of_images')
dir_path_augmented_sets_of_images = \
            os.path.join(data_root,'train_augmented_sets_of_images')
item_serial_num=0
for itemCount in itemLabel_itemCount_pairs.values:
    print("\nItem Sl. No.: ", item_serial_num)
    item_label = itemLabel_itemCount_pairs.index[item_serial_num]
    
    subdir_path_initial_set_of_images_of_item = \
                os.path.join(dir_path_initial_sets_of_images,\
                             item_label)
    subdir_path_augmented_set_of_images_of_item = \
                os.path.join(dir_path_augmented_sets_of_images,\
                              item_label)
    if not os.path.exists(subdir_path_augmented_set_of_images_of_item):
        os.makedirs(subdir_path_augmented_set_of_images_of_item)
        
    #nb of additional images per image for this class label in order to make 
    #item_serial_num size ~= AUGMENTED_ITEM_COUNT
    # 
    augmentation_ratio_for_each_image_of_this_item = \
                math.floor(AUGMENTED_ITEM_COUNT/itemCount)-1
    print("Augmenting images for: ", item_label)
    augmented_image_count = 0
    for file in os.listdir(subdir_path_initial_set_of_images_of_item):
        img=load_img(os.path.join(subdir_path_initial_set_of_images_of_item,file))
        no_of_files_in_subdir = \
            len([name for name in \
                 os.listdir(subdir_path_augmented_set_of_images_of_item)])
        
        x=img_to_array(img) 
        x=x.reshape((1,) + x.shape)
        i=0
        #https://keras.io/preprocessing/image/#imagedatagenerator-methods
        '''
DEFECTS noticed after checking the output in the subfolders
    1. Same prefix occurs multiple times. Indicates the actual batch size is
    > 1.
    2. Some prefixes do not occur at all. However,
        1. A valid batch object is however returned.
        2. A file is written
    3. In some cases the prefix does not occur, but at least one file is 
    written, that results in the file count increasing. 
    4. In some cases, however, no file is apparently written. This results in
    the file count remaining the same. My HYPOTHESIS is that a file gets 
    overwritten. The reason:
        1. The prefix_num string was introduced so as to make the file names
        unique. However, if the prefix is re-used then, since .flow on its own
        frequently generates duplicate names (that being the reason for 
        introducing prefix numbers), duolicate names would likely be generated.
        That would cause files to be overwritten.
        2. Running without the prefix will illustrate how frequently 
        overwrites happen.
    
    AN ILLUSTRATION
    ---------------
    (12:35:10) rm@ubuntu:~/.../train_augmented_sets_of_images$ find  -iname '10260*.*'
    ./rice/10260_0_6325.jpg
    ./rice/10260_0_5923.jpg
    ./rice/10260_0_8729.jpg
    ./rice/10260_0_2881.jpg
    ./rice/10260_0_992.jpg
    ./rice/10260_0_689.jpg
    (12:36:06) rm@ubuntu:~/../train_augmented_sets_of_images$ find  -iname '10261*.*'
    (12:36:14) rm@ubuntu:~/../train_augmented_sets_of_images$ find  -iname '10262*.*'
    (12:36:19) rm@ubuntu:~/../train_augmented_sets_of_images$ find  -iname '10263*.*'
    (12:36:24) rm@ubuntu:~/../train_augmented_sets_of_images$ find  -iname '10264*.*'
    (12:36:32) rm@ubuntu:~/../train_augmented_sets_of_images$ find  -iname '10265*.*'
    
        The prefixes 10261 thru 10265 have not been used. Instead 10260 has been
        reused.
        
    (12:36:54) rm@ubuntu:~/../train_augmented_sets_of_images$ find  -iname '10266*.*'
    ./rice/10266_0_1578.jpg
    ./rice/10266_0_7160.jpg
    ./rice/10266_0_9846.jpg
    ./rice/10266_0_7204.jpg
    ./rice/10266_0_2343.jpg
    ./rice/10266_0_6442.jpg
    (12:37:00) rm@ubuntu:~/../train_augmented_sets_of_images$ 
    
'''
        for batch in datagen.flow(x, \
                                  batch_size=1,\
                                  save_to_dir=subdir_path_augmented_set_of_images_of_item, \
                                  save_prefix=str(prefix_num), \
                                  save_format='jpg'):
            prefix_num += 1
            assert(1 == len(batch))
            assert(x.shape == batch.shape)
#            np.testing.assert_array_equal(x, batch) 
# assert fails. Suspect batch is the transformed version.

            i+=1; augmented_image_count += len(batch)
            current_num_of_files = \
                len([name for name in \
                     os.listdir(subdir_path_augmented_set_of_images_of_item)])
#            assert(not(len(batch) == (current_num_of_files - \
#                   no_of_files_in_subdir)))
            if(not(len(batch) == (current_num_of_files - \
                   no_of_files_in_subdir))):
                print("Difference is: ", (current_num_of_files - \
                                          no_of_files_in_subdir), \
                                    ", ",\
                                    "Prefix: ", prefix_num, \
                                    end='')
            no_of_files_in_subdir = current_num_of_files
            if i > augmentation_ratio_for_each_image_of_this_item:
                break 
    print("\nitemCount: {}; augmented image count: {}; Total: {}".format(itemCount,\
                          augmented_image_count, \
                          (itemCount + augmented_image_count)))
    item_serial_num=item_serial_num+1


# ## Let's check the new distribution

# In[11]:

print()
for dir_path, dirnames, filenames in os.walk(dir_path_augmented_sets_of_images):
   no_of_images=0
   label=''
   for filename in [f for f in filenames if f.endswith(".jpg")]:
       label=os.path.split(dir_path)[1]
       no_of_images+=1

   if(not('' == label)):
       print("label: {}; no_of_images: {}".format(label,no_of_images))
       

print("\n\tDONE: ", __file__)

