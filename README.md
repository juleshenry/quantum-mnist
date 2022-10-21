# quantum-mnist
This project seeks to confirm results published in https://thesai.org/Downloads/Volume11No10/Paper_40-Handwritten_Numeric_Image_Classification.pdf

We will explore quantum image processing via plankton data set found in this paper: https://arxiv.org/pdf/2108.05258.pdfß

Other relevant research:
https://arxiv.org/pdf/2011.02831.pdf

# Phase One: Confirm research conclusions in google colab
Done. We have tested the quantum mnist colab and it works. 

# Phase Two:
Apply lessons in Phase One to the plankton data set. We will look at the cartesian product comparison of binary classification via the "fair" 4x4 neural net. Do the same for the quantum scenario.

We need to convert the plankton dataset to grayscale. We do so using the default configuration in Pillow. Next, we downsize colors to normalize. In the current iteration, the hyperparameters of batch_size and image size must be explored. The size of 4x4 is insufficient for the plankton dataset, so a larger quantum simulation will be needed than described in the 4x4 method. Even in 16x16, the naive FFN is not powerful.		
(We are here)

# Phase Three: Generalize binary quantum classifier 
In this phase, we devise a generalization that maps to images containing 0-9.

# Phase Four : Apply general quantum classifier to plankton dataset
In this phase, we perform the generalized algorithm on the plankton dataset and compare to the deep learning approach found [here](https://arxiv.org/pdf/2108.05258.pdf).
