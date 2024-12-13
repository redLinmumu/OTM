
<!-- description -->
This rep is for clustering experiment in the paper. Here are the description of main files:

fs_environment.yml is the conda  environment file 

mwae_bm.py is the main file for MWAE with OTM

mwae_single.py is the file for single modality.

compute_single_acc.py computes accuracy for single modality.

compute_cal_acc.py computes accuracy for caltech7.

<!-- directory -->

directory utils are supportive tools.

models is the directory to save models.

data is the directory for downloaded datasets.

configs are for running mwae_bm.py and mwae_single.py.

<!-- Run the multimodal clustering method -->
To run our code, please make sure you have an environment that installs required packages. We provide the fs_environment.yml of our environment for reference.

The commands are as follows:

python mwae_bm.py --cfg configs/movie_unalign/mwae_bm.yml

python mwae_bm.py --cfg configs/orl_unalign/mwae_bm.yml

python mwae_bm.py --cfg configs/pro_unalign/mwae_bm.yml


<!-- dataset -->

[notice]:

We only provide three datasets, movies, orl and prokaryotic, due to the uploading restriction on size. 
