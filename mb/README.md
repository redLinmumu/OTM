This rep is for classification and regression experiments based on framework MultiBench. We only list part code and no dataset due to the restriction of uploading. The dataset can be downloaded using the script in data or datasets folders. All source code of Multibench is available on https://github.com/pliang279/MultiBench. .



Since our code is based on Multibench in https://github.com/pliang279/MultiBench, there are some names belong to public Multibench, especially when fetching data, such as pliang, which have nothing to do with us.


Below are Colab Links to the Multibench Tutorials from Multibench:

- [Tutorial 1](https://colab.research.google.com/github/pliang279/MultiBench/blob/main/examples/Multibench_Example_Usage_Colab.ipynb): This example shows a very basic usage case of MultiBench. In particular, it demonstrates how to use MultiBench with the affective computing dataset MOSI, and how to use it with a very simple fusion model.
- [Tutorial 2](https://colab.research.google.com/github/pliang279/MultiBench/blob/main/examples/Multibench_Example_Usage_On_Colab_Part_2_MFAS.ipynb): This example shows a slightly more complicated training paradigm in MultiBench - MFAS (MultiModal Fusion Architecture Search) model on the AVMNIST dataset.
- [Tutorial 3](https://colab.research.google.com/github/pliang279/MultiBench/blob/main/examples/Multibench_Example_Usage_On_Colab_Part_3_MCTN.ipynb): This example shows a slightly more complicated training paradigm in MultiBench. Namely, we'll run MCTN (learning representations by translating from one modality to another) on MOSI.


Our code structure is consistent to Multibench. 

<!-- directory description -->
best_pts is the directory that owns the best models of classification and regression tasks, including the missing modality case.  This folder can be used in the missing modality experiments. 

data is the directory for data. The way to get the data is provided by multibench in public. 

datasets is the directory for constructing dataloader. We reconstruct the dataloader with 5-nfold cross-validation. 

examples is the directory of our experiments codes mostly. Based on the names of the files, they are easy to detect  the corresponding experiment scenarios. CMU-MOSI is under directory affect/, mujoco is  under directory gentle_push/, enrico is  under directory hci/, and av-mnist is under directory multimedia. For mujoco, we run the data preprocessing code in the first place, and then use the cached files directly next time to speed up. 

fusions is  the directory of fusion strategies(from MultiBench, mostly).

mwae is  the directory of OTM implementation.

objective_functions is  the directory of objective implementation(from MultiBench, mostly). 

private_test_scripts is  the directory of test implementation(from MultiBench, mostly). 

training_structures is the directory of the supervised learning. We reconstruct the trainning progress for OTM.

unimodals is the directory of models for single modality(from MultiBench, mostly).

utils is the directory of parse args or others.


<!-- env -->
mmb_enviroment.yml is the conda export file of our environment.

Before running our code,  please make sure that you have a suitable environment that installs requrired packages, and our yml file is for reference.



[Notice]:

Some Multibench codes are not included in our rep, and they can be found in the Multibench rep, and we just remove them for the uploading  size restrictions. 
