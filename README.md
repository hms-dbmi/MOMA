# MOMA: The Multi-omics Multi-cohort Assessment (MOMA) Platform
Histopathology Images Predicted the Key Multi-Omics Aberrations and Prognoses in Colorectal Cancer: A Multi-Cohort Validation Study (in preparation)

![](https://i.imgur.com/qm4OLtI.png)



## Requirements
* Survival Prediction
    * Python==3.6.0
    * tensorflow==2.4.0
    * lifelines
    * scipy
    * statistics
    * matplotlib
* Multi-omics characterization
    * Python==3.6.0
    * torch==1.6.0
    * torchvision==0.7.0
    * sciki-learn
    * numpy
    * [smooth-topk](https://github.com/oval-group/smooth-topk)
    * opencv-python
    * tqdm

## Data Proprocessing
Please refer to this github [repo](https://github.com/mahmoodlab/CLAM), or you can download the processed dataset provided by [Kather et al](https://www.nature.com/articles/s41591-019-0462-y).
Please refer to this github [repo]([https://github.com/mahmoodlab/CLAM](https://github.com/BMIRDS/deepslide)])

## Feature Extraction
You can use any pre-trained CNN model (like our multi-omics characterization task) or train model on our own (like our survival prediction task) to extract each patchs' features.

## Data Preparation
* Survival Prediction
    * Color normalization
    * Make a dataframe
        ``` python
        # Survival dataframe
        data = {
            'bcr_patient_barcode' : patient id,
            'vital_status' : overalall survival status or disease free status,
            'Days' : overalall survival days or disease free days
            '0' : pathology image feature (dimension 1)
            '1' : pathology image feature (dimension 2)
            ...
            'n' : pathology image feature (dimension n)
        }
        
        df = pd.DataFrame(data)
        ```

* Multi-omics characterization

    * XXX_id can be patient’s ID or slide’s ID, which is depanding on your task. And please be sure that the patch_name in features pickle file and in cluster pickle file is the same.
    * Sample file

        ``` python
        # Patch features pickle
        {
          'patch_name' : array([latent feature]),
          'patch_name' : array([latent feature]),
          ...
        }
        ```

        ``` python
        # Cluster pickle file
        {
          XXX_id: {
            'patch_name' : cluster label,
            'patch_name' : cluster label,
            ...
          },
          XXX_id: {
            'patch_name' : cluster label,
            'patch_name' : cluster label,
            ...
          },
        }

        ```

        ``` python
        # Label pickle file
        {
          XXX_id: class,
          XXX_id: class,
          ...
        }
        ```



## Usage
* Survival Prediction
    * Both Overall survival prediction and disease free prediction are the same .ipynb file

* Multi-omics characterization
    * Sample Command
        ``` python
        # Training
        python3 Train.py --level patient --hidden_dim 512 --encoder_layer 6 --k_sample 3 --tau 0.5 --save_path 'path/to/save/' --label 'path/to/label pickle file' --use_kather_data True --epoch 60 --lr 3e-4 --evaluate_mode kfold --kfold 5
        ```
        ``` python
        # Validation
        python3 Validation.py --level patient --hidden_dim 512 --encoder_layer 6 --k_sample 3 --tau 0.5 --save_path 'path/to/save/' --label 'path/to/label pickle file' --use_kather_data True
        ```
        ```shell script

        --level                 slide or patient level
        --hidden_dim            The dimension in the Transformer encoder
        --encoder_layer         The layers of the Transformer encoder
        --k_sample              The top-k and bottom-k for the instance selection
        --tau                   The smoothness term for smoothSVM
        --use_kather_data       Using the data provided by kather et al. or not
        --save_path             Model weights save path
        --label                 Path to label pickle file
        --lr                    Learning rate
        --epoch                 Training epochs
        --evaluate_mode         Kfold or holdout test
        --kfold                 The number of fold
        ```

