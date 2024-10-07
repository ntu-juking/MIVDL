# Empowering Distributed Learning-based Vulnerability Detection via Multi-Modal Prompt Tuning

This is MIVFL vulnerability detection.

If you want to reproduce the program.
### 1 please move to 'slicing' dir.
Reproducing the slicing needs to download 'Joern', we recommend version 1.0. 
We have placed the specific steps for reproducing in the 'slicing' directory.

### 2 move to 'processing' dir.
The order of executing files is as follows:

1、input.py

2、extract.py

3、data.py

4、script.py

Among them, 'remove' file is used to remove files, and word2vec is used to train word vector models. You can also choose not to train the word vector model. Please follow the README file in the 'processing' directory for specific operations on other files

### 3 move to 'feature_extraction' dir.
The content in this directory is mainly responsible for extracting the relevant features of vulnerability code. Execute the main. py file, the required parameters are listed in the directory.

### 4 move to 'interactive' dir.
Execute the main.py file.

Please note that the model here requires downloading graphcodebert-base, please refer to the link: https://huggingface.co/transformers/v4.2.2/index.html。
