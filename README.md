# Empowering Distributed Learning-based Vulnerability Detection via Multi-Modal Prompt Tuning

## Introduction
Our approach, **MIVDL**, demonstrates that combining distributed learning with multi-modal prompt tuning significantly enhances vulnerability detection performance. Key highlights of **MIVDL** include:

+  **Privacy-Preserving Distributed Learning:** The method uses distributed learning to enable local models to interact without sharing data, ensuring privacy and security.

+  **Multi-Modal Code Analysis:** It integrates structured (e.g., Code Property Graphs) and unstructured (e.g., source code tokens) data, capturing diverse semantic and structural features for better detection accuracy.

+  **Hybrid Representation:** Combines multiple code modalities into a unified representation using pre-trained models, leveraging their unique strengths.


If you want to reproduce the program, please follow these steps:
### 1. Move to 'slicing' dir.
To reproduce the slicing, you must download 'Joern'; we recommend version 1.0 for "Joern". 
We have placed the specific steps for reproducing in the 'slicing' directory.

### 2. Move to 'processing' dir.
The order of executing files is as follows:

step 1: run ``` input.py ```

step 2: run ``` extract.py ```

step 3: run ``` data.py ```

step 4: run ``` script.py ```

For the rest files, the ```remove file.py``` is used to remove specific files, and ```word2vec.py``` is used to train word-to-vector models. You can also choose not to train the word vector model. 

### 3. Move to 'feature_extraction' dir.
The content in this directory is mainly responsible for extracting the relevant features of vulnerability code. To do so, execute the ```main.py``` file. The required parameters are listed in the current directory.

### 4. Move to 'interactive' dir.
Execute the ```main.py``` file.

Note that the model here requires downloading GraphCodeBERT-base. Please refer to this [Huggingface url-link](https://huggingface.co/transformers/v4.2.2/index.html).
