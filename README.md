# GATAERP
This is the  code of paper “GATAERP: A Graph Attention Autoencoder for Relationship Prediction of Cancer Progression”

Guide Steps:

1、Download source codes from github

2、Install anconda , prepare python environment and cuda environment. 

A nvidia graphics card is not required, but it helps if you have one.

3、Install requires from setup.py( by conda install or pip install)

some commands:

  pip install rpy2
  
  conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
  
  conda install matplotlib==3.5.1 (I found that the latest matplotlib has bugs)
  
4、Run codes from main.py  ggae_predict function can show the result of k folds.

5、If you want to run dynoverse you need install docker(it may need 8G RAM) and r environment on your PC.

visit  https://dynverse.org/ for dynoverse guidence.

If you have any questions please ask in Issues.

We are glad to answer your questions and take your suggestions.

