# Intel-Image_Classification
Image Classification on Intel Image Classification dataset

The dataset contain around 14K training images and 3K test images. The input belongs to either one of the six categories 

 1. buildings
 2. forest
 3. glacier
 4. mountain
 5. sea
 6. street

The model achieves just over 80% validation accuracy. Tried another iteration with data augmentation, which  resulted in lower accuracy for this particular dataset. The training and validation losses for various epochs are shown below:

![Iter1_plot](https://user-images.githubusercontent.com/20210669/103241065-ff52f880-4949-11eb-8697-78c1f34c9caf.png)

The plot reveals that the model starts overfitting in the early epochs. Different ways to improve the model:

 - Using pretrained network such as Xception, Resnet50, VGG16 etc.
 - Fine tuning the parameters
 - Callback options such as  'ReduceLROnPlateau' in tensorflow keras
 
