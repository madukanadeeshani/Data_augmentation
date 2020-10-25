import glob
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 
   
''' Initialising the ImageDataGenerator class. 
We will pass in the augmentation parameters in the constructor. '''
datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5)) 
    
# Loading a sample image  
#img = load_img('E:/AUTISUM_RESEARCH/Data_augmentation/new/*.jpg')
list_of_files = glob.glob('E:/AUTISUM_RESEARCH/Data_augmentation/negative_jpg/*.jpg')
#print(list_of_files)
for file in list_of_files:
    img = load_img(file)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1, ) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size = 1, save_to_dir ='E:/AUTISUM_RESEARCH/Data_augmentation/augmented_negative',  save_prefix ='negative', save_format ='jpg'):
    #x += 1
        i += 1
        if i > 5: 
            break
 
# Converting the input sample image to an array 
#x = img_to_array(img) 
# Reshaping the input image 

   
# Generating and saving 5 augmented samples  
# using the above defined parameters.  
