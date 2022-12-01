# Image Segmentation on small subset of Cityscapes


Download the dataset here: 

**Cityscapes: Semantic Understanding of Urban Street Scenes.** https://www.cityscapes-dataset.com/downloads

Given that these are large files of several or more Gigabytes, it is important for the practitioner to judiciously select only the data needed for the specific task of interest. The focus of this study is to test the performance of different models for semantic segmentation, so the following datasets (file size in paratheses) were downloaded: **gtFine_trainvaltest.zip** (241MB) and **leftmg8bit_trainvaltest.zip** (11GB). 

The gtFine_trainvaltest.zip dataset contains multiple annotations. Of these files, *gtFine_labellds were the files used for labels and also for creating the segmentation masks. 

The leftmg8bit_trainvaltest.zip dataset contains the original 2048x1024 images of the different street scenes with extension names *leftlmg8bit.png. 

(For this task, object detection and classification by color were not explored so the *gtFine_polygons.json and *gtFine_color.png files were never used, nor were the *gtFine_instancelds.png files since we focus only on semantic segmentation in this current code.)

After unzipping these files, you will find the subdirectories: **train**, **val**, and **test**. We focus on the files contained in train and val folders.

From these datasets, a smaller subset was created from the Hamburg data (in train/Hamburg/) and Frankfurt data (in val/Frankfurt/). 

 Your directory structure should look like this (for logging later on, optionally add a folder called myfcn_test or whichever name you prefer.)

```
SemanticSegmentationCityscapes/

   myfcn_test/
   data/
      train/
         (files from train/Hamburg folder with the following extensions:)
         *gtFine_labellds.png
         *leftlmg8bit.png
      valid/
         (files from val/Frankfurt folder with the following extensions:)
         *gtFine_labellds.png
         *leftlmg8bit.png
   seg_code/
      __init__.py
      models.py
      train.py
      utils.py
      special_transforms.py
   ```
 ---------------------------------------------------------------------------------------
## Objectives of this code

This code is simplified and designed for running on modest resources. The computational details are summarized in the table below:

Table 1. Computational Methods
_______ | ______________
1 GPU	 | NVIDIA GeForce GTX 1660 Ti
_______ | ______________
Framework	| PyTorch 1.8.1 (Paszke & et al, 2019)
_______ | ______________
Number of Epochs |	500 
_______ | ______________
Batch Size	| 8 
_______ | ______________
Learning rate	| 1x10-3
_______ | ______________
Optimizer	| Adam optimizer with weight decay of 1 x10-5 (Kingma & Ba, 2014) 
_______ | ______________
Criterion	| Cross Entropy Loss with Focal Loss (Lin et al., 2018)
_______ | ______________
Models	| my_FCN – built-from-scratch UNET (this work), U-Net – (Ronneberger et al., 2015), DeepLab+ (ResNet101) – (L.-C. Chen et al., 2018; He et al., 2016)
_______ | ______________
Dataset	| Cityscapes dataset (using only the Hamburg and Frankfurt subsets) (Cordts, 2022; Cordts et al., 2016)
_______ | ______________
Evaluation Metrics |	Global accuracy, average accuracy and Intersection-over-Union (IoU)
	_______ | ______________



We want to 
1. Visualize the dataset
2. Practice data pre-processing techniques (data augmentation)
3. Train an image segmentation network to accurately classify objects appearing in the street scenes (cars, footpath, road, pedestrians, etc.). 

_________________________________________________________________________________________
## Setting up your environment with Conda

Install dependencies using:

<code> python -m pip install -r requirements.txt </code>

and if you are using conda use:

<code> conda env create torch_segment.yml </code>

which can be activated for your Python environment using: 

<code> conda activate torch_segment </code>

--------------------------------------------------------------------------------------------------------
You can train the models using the shallow network, my_FCN, using the following command:

<code> python -m seg_code.train -m my_fcn </code>

To train the reference UNet, the code is: 

<code> python -m seg_code.train -m unet </code>

To train the DeepLab (with pretrained ResNet-101), the code is: 

<code> python -m seg_code.train -m deeplab </code>

---------------------------------------------------------------------

To do some visualization:

Here is the code you can run before training the model to see a snapshot of what the dataset looks like:

<code> python -m seg_code.utils  </code>

_____________________________________________________________________
If you want to use Tensorboard, here is some extra code:

<code> python -m seg_code.train -m my_fcn --log_dir myfcn_test -n 1000 </code>

followed by:

<code> tensorboard --logdir=myfcn_test --port 6006 --bind_all  </code>
             
the message you'll receive will give you something like:

<code> http://your-Laptop-name:6006/ </code>

click on the address you get and open it in a web browser. See the interactive tensorboard. Done!

____________________________________________________________________________________
## Most Helpful References:

Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ArXiv. https://arxiv.org/pdf/1802.02611.pdf

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention, 234–241.

jfzhang95. (2018). PyTorch DeepLab-XCeption. GitHub. https://github.com/jfzhang95/pytorch-deeplab-xception

milesial. (2021). Pytorch-Unet. Github. https://github.com/milesial/Pytorch-UNet

Sai Ajay Daliparthi, V. S. (2021a). The Ikshana Hypothesis of Human Scene Understanding. Github. https://github.com/dvssajay/The-Ikshana-Hypothesis-of-Human-Scene-Understanding

