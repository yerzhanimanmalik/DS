## Neuroml

Repository includes neuroml course project:

Comet visualizations:   https://www.comet.ml/imanmal1k/vae-clusterization/view/MA2Rpx2ebdlbMa4IxRmrjduKZ


## Project plan and report:

(50%) 1. Planned: Implement 3d Variational Autoencoder   . +  Done: Implemented 3d Autoencoder

(65%) 2. Planned: Obtain latent vectors for fmri task and rest scans    . +     Done

(90%) 3. Planned: Clusterize latent vectors into task and rest    .  +      Done: Classification, since we know true labels (task and rest). 

         P.S. Term clusterization is more apt to the general problem, where we are differentiating brain connectivity networks (e.g. Broca, motor etc)
         
(100%) 4. Planned: Decode avg latent vectors from a group. Calculate validity of classification        .        +/- Done: Calculated accuracy of the classifier


## Experiment description:

- Dataset: 40 patients from la5 dataset. 20 for task and 20 for resting state fmri activations.
- Each rest activation is 152 time stamps, each task activation is 267 time stamps
- 3D shape of mri scans is cropped from   (65, 77, 49)   to   (64, 64, 48)

- Autoencoder architecture:
         self.encoder = nn.Sequential(
                nn.Conv3d(1, 32, 4, 2, 1, bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 64, 4, 2, 1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.Conv3d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
                nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(64, 32, 4, 2, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(32, 1, 4, 2, 1, bias=False),
                nn.ReLU(inplace=True)
        )  
    - Loss: MSE
    - Optimizer: Adam (lr=1e-2)
    - N_epochs: 10
        
- Classifier architecture:
         self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12288, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
     - Loss: BCEWithLogits
     - Optimizer: Adam (lr=1e-2)
     - N_epochs: 10
     
        
 - Experiments conducted were performed using Colab and Azure gpu. Calculations were done for 10 epochs.


## Summary:

The goal of encoding full-size fmri scans of task and rest la5 data has been achieved. Later on, latent vectors has been used to train the fully-connected network to classify if the original scan belongs to task or rest activations. After training over 16 hours (for both encoder and classifier) model has achieved 64% accuracy.


## Ideas for future improvements:

1. Use Variational autoencoder instead of a simple autoencoder

2. Use temporal structure of fMRI scans by applying RNN or even more complex models, like LSTMs or GRU

3*. Make clusterization of latent vectors. Find correspondence between obtained clusters and real (labeled) connectivity networks

