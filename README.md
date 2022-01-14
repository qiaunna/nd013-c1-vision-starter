# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
Object Detection for self-driving cars is an important task for autonomous navigation. The embedded software must be accurate and reliable for real-time computer vision image processing. In this project, we will build a Convolutional Neural Network (CNN) containing multiple layers that filter and map pixels. The image will pass through a series of convolution layers that compare small squares of input data from the image to detect 3 classes: vehicles, pedestrians, and cyclists using the Waymo Open Datset. The preprocessed data is provided in the Udacity workspace: /home/workspace/data/waymo/training_and_validation. The test data contains 3 tfrecord files the location: /home/workspace/data/waymo/test. The project explores splitting data for an efficient model and visualizing object detection with bounding boxes. The Waymo Open Dataset contains a diverse set of images to be visualized in different weather conditions and scenerios.

### Set up
Please follow the readme instructions for local setup. The Udacity Virtual Machine has all the necessary dependencies and extensions installed.

### Dataset
#### Dataset analysis
The Waymo Open Dataset was used to train a neural network model. The data within the Udacity classroom has 97 tfrecord available for training. These tfrecord files contain 1 frame per 10 seconds from a 10fps video. These images are annotated bounding boxes for 3 classes (vehicles, pedestrian, cyclists). The images from the tfrecord include:

(Foggy)
![2](https://user-images.githubusercontent.com/22205974/149421930-f56b7ce1-ff41-417c-80a9-2898f4c22b12.PNG)

(Image noise/blurry)
![3](https://user-images.githubusercontent.com/22205974/149421951-fc012e54-a1e5-4db8-82b4-a69f05286b2e.PNG)

(Night)
![4](https://user-images.githubusercontent.com/22205974/149421965-553a2ba7-a8a4-462a-a6dd-b5b7666d2cf2.PNG)

(Good weather conditions)
![5](https://user-images.githubusercontent.com/22205974/149421980-978a3123-1909-4ec4-aca3-0101f457d0d8.PNG)
![7](https://user-images.githubusercontent.com/22205974/149421996-bbf23b21-0f15-4438-a54d-93517ecb7a6c.PNG)
![8](https://user-images.githubusercontent.com/22205974/149422002-bc4fa1c1-ad16-421b-a0d2-4f43ca75fd37.PNG)

(Multi-Classes tracked)
![1](https://user-images.githubusercontent.com/22205974/149421898-077ca256-c614-447a-ae21-84eb239b15b9.PNG)
![6](https://user-images.githubusercontent.com/22205974/149421988-345f8246-1aa9-4f30-86ab-4f470eb30863.PNG)
![9](https://user-images.githubusercontent.com/22205974/149422058-f2e7325e-d055-47f6-bc43-fe5c13806091.PNG)
![10](https://user-images.githubusercontent.com/22205974/149422066-9c52d2bf-ef8a-4f36-9c1a-35bf460166c2.PNG)


The Single Shot Detector Model is an object detection algorithm that was used to train the Waymo Open Dataset. This model detected 3 classes from the dataset: vehicles, pedestrians, and cyclist. The frequency distribution of these classes are based on the analysis of 1000 and 10,000 shuffled images in the training dataset. In 1,000 images 76% of vehicles, 24% of pedestrians and less that 1% were cyclists were tracked. This produced very few shuffled images containing cyclists.

![waymo1000](https://user-images.githubusercontent.com/22205974/149423493-73f41e24-476b-40db-a9b5-8b78cb387c5a.PNG)

In 10,000 images 75% of vehicles, 24% of pedestrians and 1% were cyclists were tracked. This increseased the number of cyclists tracked from the dataset.
![waymo10000](https://user-images.githubusercontent.com/22205974/149423514-2443ea05-dc4f-437a-a48f-6e0428b38fca.PNG)


#### Cross validation
97 tfrecord files were split 85:15, 82 files for training and 15 files for validation. The testing file contains 3 tfrecord file preloaded into the Udacity workspace. In order to properly train the neural network the image were shuffled for cross validation. The create_splits.py shuffles the files before splitting the dataset. Shuffling the data helps the algorithm avoid any bias due to the order of the data was assembled. These bias would cause the algorithm, to visualize patterns read from the previous images that may not be in the following images and  this would cause overfitting.

### Training
#### Reference experiment
The Single Shot Detector (SSD) Resnet 50 model was used to pretrain our dataset. The SSD model has two image augmentations present, random horizontal flip and random image crop. In the images below the training loss is highlighted in orange and the validation loss in blue. The training and validation loss for the reference model without adding any extra augmentations are displayed. A learning rate of .04, on the initial experiment resulted in a low learning rate. A learning rate that is rapidly decreased to a minimum value before increasing rapidly again is desired. After multiple training iterations the final training rate was set at 0.0001819.


![1loss](https://user-images.githubusercontent.com/22205974/149424941-c1ad1bc4-fa6c-4578-a4db-5482c90d4ce6.PNG)
![2loss](https://user-images.githubusercontent.com/22205974/149424946-421549f6-cbdb-4ac9-a3b0-77ab49caa8da.PNG)
![3loss](https://user-images.githubusercontent.com/22205974/149424963-00b0cb00-5b9c-4e5c-9ae8-8a6bf3c20ea1.PNG)
![4loss](https://user-images.githubusercontent.com/22205974/149424985-183aeaac-a773-45ff-bfad-70d7cd27a83e.PNG)
![5loss](https://user-images.githubusercontent.com/22205974/149425140-4ef3778e-d146-45bd-bfc7-3c51971a7f6e.PNG)


This model performs poorly on the training and validation dataset.


#### Improve on the reference
To improve object detection within the model: the learning rate can be increased or decreased, the training steps can be increased and augmentations can be added to the images. A new version of the config file is created (pipeline_new.config) and contains modifications to improve the model performances.


#### Image Augmentations
The following aumentations were applied to the images below:

The image was flipped (random_horizontal_flip): This presents a mirrored image that helps to train the model to recognize objects in the opposite direction.

![aug1](https://user-images.githubusercontent.com/22205974/149422482-e5528a26-fd91-495e-bacf-bc847fa214a1.PNG)
![aug10](https://user-images.githubusercontent.com/22205974/149422594-59c84061-0624-4b34-ba0d-2f8bc96e9094.PNG)

The image was randomly cropped using:

}

  data_augmentation_options {
  
    random_crop_image {
    
      min_object_covered: 0.0
      
      min_aspect_ratio: 0.75
      
      max_aspect_ratio: 3.0
      
      min_area: 0.75
      
      max_area: 1.0
      
      overlap_thresh: 0.0
      
    }
    
    
![aug2](https://user-images.githubusercontent.com/22205974/149422486-b610f83f-b51f-4078-97b6-14711a4ab240.PNG)
    
    

The image was converted into grayscale (random_rgb_to_gray) 0.03 probability. RGB images need 24 bits for processing while grayscale only needs 8bits to process. Grayscale provides faster processing times and helps with feature distinction.

![aug3](https://user-images.githubusercontent.com/22205974/149422503-0257b2d3-b2ae-4f62-8e16-80a61274a425.PNG)
![aug5](https://user-images.githubusercontent.com/22205974/149422526-7b45c568-b458-4cbe-9a9d-fd07f98de7b9.PNG)

The image was augmented to adjust the brightness by delta 0.3. Over exposure to light can make it harder for the model to distingush the objects features.

![aug9](https://user-images.githubusercontent.com/22205974/149422578-429c7d88-771f-47b4-aa49-297a75904ca4.PNG)
![aug6](https://user-images.githubusercontent.com/22205974/149422535-eedf1769-fcd3-4fcd-b6d8-c07e3c1b0fce.PNG)

The image was augmented to the contrast. Training the model with darker images can provide a better model for object recognition in darker images.
![aug4](https://user-images.githubusercontent.com/22205974/149422512-0227197c-3a91-4984-a2ac-5ab1047b694a.PNG)
![aug8](https://user-images.githubusercontent.com/22205974/149422564-026733eb-fcf9-4765-a5eb-246a9dd840f8.PNG)
![aug7](https://user-images.githubusercontent.com/22205974/149422550-321536ba-c66c-4cdc-a4f7-80822991bc10.PNG)


#### Adjust Learning Rate
The following images are the loss metrics for experiment 2.

![alr1](https://user-images.githubusercontent.com/22205974/149428180-58f29b54-cddc-4297-bb05-8c6a320e7eb5.PNG)
![alr2](https://user-images.githubusercontent.com/22205974/149428198-5c7bb181-1d49-47bb-86e1-fd38379f3502.PNG)
![alr3](https://user-images.githubusercontent.com/22205974/149428211-5c5757c4-a498-4a6c-ad71-fc02410462fb.PNG)

![alr4](https://user-images.githubusercontent.com/22205974/149428218-5f251d54-fda4-4b13-8b8d-49007c3cf972.PNG)
![alr5](https://user-images.githubusercontent.com/22205974/149428224-567a2a92-a015-44b5-81a6-b699182c70b0.PNG)

This model performs the best in detecting the density of tracked vehicles in the dataset. The mean average precision for large boxes in the figure below is 0.5 and reduces to approximately .2 for small tracked objects. The adjusted learning rate for the augmented model is displayed in the image below. This model performs considerably better than the first model and is evident by the loss metrics. The model tracks vehicles in a variety of weather conditions and augmentations. The dataset is unbalanced with a multitude of vehicles present, while providing less datapoints for pedestrians and cyclists. The dataset under performs in detecting the pedestrian and cyclist class. The model can be trained better to track all classes if the classes to be track provide a better balance.

![exp_2_map](https://user-images.githubusercontent.com/22205974/149428911-2a311624-f194-4710-8347-bf6c1536b8b7.png)
