# Empowering Distributed Learning-based Vulnerability Detection via Multi-Modal Prompt Tuning

Our approach, MIVDL, demonstrates that combining distributed learning with multi-modal prompt tuning significantly enhances vulnerability detection performance.

If you want to reproduce the program.
### 1. Please move to 'slicing' dir.
To reproduce the slicing, you must download 'Joern'; we recommend version 1.0 for "Joern". 
We have placed the specific steps for reproducing in the 'slicing' directory.

### 2. Move to 'processing' dir.
The order of executing files is as follows:

step 1: run ``` input.py ```

step 2: run ``` extract.py ```

step 3: run ``` data.py ```

step 4: run ``` script.py ```

Among them, the 'remove' file is used to remove files, and word2vec is used to train word-to-vector models. You can also choose not to train the word vector model. Please follow the README file in the 'processing' directory for specific operations on other files.

### 3. Move to 'feature_extraction' dir.
The content in this directory is mainly responsible for extracting the relevant features of vulnerability code. Execute the ```main.py``` file. The required parameters are listed in the current directory.

### 4. Move to 'interactive' dir.
Execute the ```main.py``` file.

Note that the model here requires downloading GraphCodeBERT-base. Please refer to this [Huggingface url-link](https://huggingface.co/transformers/v4.2.2/index.html).
