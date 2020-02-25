# Multimodal classification of social media content

Multimodal classification for social media content is an important problem. There is also a lack of resources. The idea here is to train a basic deep learning based classifiers using one of the publicly available multimodal dataset.

## Download dataset:
Before trying to start running any script, please download the dataset first. More detail of this dataset can be found here: https://crisisnlp.qcri.org/crisismmd.html and the associated published papers.

* Download the dataset (https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz)

Assuming that your current working directory is YOUR_PATH/multimodal_social_media
```
tar -xvf CrisisMMD_v2.0.tar.gz
mv CrisisMMD_v2.0/data_image $PWD/
```

* Download the word2vec model and place it under your home or current working directory, (https://crisisnlp.qcri.org/data/lrec2016/crisisNLP_word2vec_model_v1.2.zip)

You need to modify the word2vec model path in ```bin/text_cnn_pipeline_unimodal.py``` script. 

## Install dependencies:
python 2.7

#### Create a virtual environment
```
python -m venv multimodal_env python=2.7
```
#### Activate your virtual environment
```
source $PATH_TO_ENV/multimodal_env/bin/activate
```

#### Install dependencies
```
pip install -r requirements_py2.7.txt
```

## Run unimodel classifiers:

### Unimodel - text

```bash
CUDA_VISIBLE_DEVICES=1 python bin/text_cnn_pipeline_unimodal.py -i data/task_data/task_informative_text_img_agreed_lab_train.tsv -v data/task_data/task_informative_text_img_agreed_lab_dev.tsv -t data/task_data/task_informative_text_img_agreed_lab_test.tsv \
--log_file snapshots/informativeness_cnn_keras.txt --w2v_checkpoint w2v_checkpoint/word_emb_informative_keras.model -m models/informativeness_cnn_keras.model -l labeled/informativeness_labeled_cnn.tsv -o results/informativeness_results_cnn.txt >&log/text_info_cnn.txt &

CUDA_VISIBLE_DEVICES=0 python bin/text_cnn_pipeline_unimodal.py -i data/task_data/task_humanitarian_text_img_agreed_lab_train.tsv -v data/task_data/task_humanitarian_text_img_agreed_lab_dev.tsv -t data/task_data/task_humanitarian_text_img_agreed_lab_test.tsv \
--log_file snapshots/humanitarian_cnn_keras.txt --w2v_checkpoint w2v_checkpoint/word_emb_humanitarian_keras.model -m models/humanitarian_cnn_keras.model -l labeled/humanitarian_labeled_cnn.tsv -o results/humanitarian_results_cnn.txt >&log/text_hum_cnn.txt &

```
### Unimodel - image

```bash
CUDA_VISIBLE_DEVICES=0 python bin/image_vgg16_pipeline.py -i data/task_data/task_informative_text_img_agreed_lab_train.tsv -v data/task_data/task_informative_text_img_agreed_lab_dev.tsv -t data/task_data/task_informative_text_img_agreed_lab_test.tsv  \
-m models/informative_image.model -o results/informative_image_results_cnn_keras.txt >& log/informative_img_vgg16.log &

CUDA_VISIBLE_DEVICES=1 python bin/image_vgg16_pipeline.py -i data/task_data/task_humanitarian_text_img_agreed_lab_train.tsv -v data/task_data/task_humanitarian_text_img_agreed_lab_dev.tsv -t data/task_data/task_humanitarian_text_img_agreed_lab_test.tsv \
-m models/humanitarian_image_vgg16_ferda.model -o results/humanitarian_image_vgg16.txt >& log/humanitarian_img_vgg16.log &

```


## Run multimodel classifiers:

```bash
# convert images to numpy array
python bin/image_data_converter.py -i data/all_images_path.txt -o data/task_data/all_images_data_dump.npy

CUDA_VISIBLE_DEVICES=1 python bin/text_image_multimodal_combined_vgg16.py -i data/task_data/task_informative_text_img_agreed_lab_train.tsv -v data/task_data/task_informative_text_img_agreed_lab_dev.tsv \
-t data/task_data/task_informative_text_img_agreed_lab_test.tsv -m models/info_multimodal_paired_agreed_lab.model -o results/info_multimodal_results_cnn_paired_agreed_lab.txt --w2v_checkpoint w2v_checkpoint/data_w2v_info_paired_agreed_lab.model --label_index 6 >& log/info_multimodal_paired_agreed_lab.log &

CUDA_VISIBLE_DEVICES=0 python bin/text_image_multimodal_combined_vgg16.py -i data/task_data/task_humanitarian_text_img_agreed_lab_train.tsv -v data/task_data/task_humanitarian_text_img_agreed_lab_dev.tsv \
-t data/task_data/task_humanitarian_text_img_agreed_lab_test.tsv -m models/hum_multimodal_paired_agreed_lab.model -o results/hum_multimodal_results_cnn_paired_agreed_lab.txt --w2v_checkpoint w2v_checkpoint/data_w2v_hum_paired_agreed_lab.model --label_index 6 >& log/hum_multimodal_paired_agreed_lab.log &

```


## Please cite the following paper if you are using the data:

* *Firoj Alam, Ferda Ofli, and Muhammad Imran, "Crisismmd: Multimodal twitter datasets from natural disasters", Twelfth International AAAI Conference on Web and Social Media. 2018.*

* *Ferda Ofli, Firoj Alam, and Muhammad Imran, "Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response", 17th International Conference on Information Systems for Crisis Response and Management, 2020.*

```bib
@inproceedings{multimodalbaseline2020,
  Author = {Ferda Ofli and Firoj Alam and Muhammad Imran},
  Booktitle = {17th International Conference on Information Systems for Crisis Response and Management},
  Keywords = {Multimodal deep learning, Multimedia content, Natural disasters, Crisis Computing, Social media},
  Month = {May},
  Organization = {ISCRAM},
  Publisher = {ISCRAM},
  Title = {Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response},
  Year = {2020}
}

@inproceedings{crisismmd2018icwsm,
  author = {Firoj Alam and Ofli, Ferda and Imran, Muhammad},
  title = {CrisisMMD: Multimodal Twitter Datasets from Natural Disasters},
  booktitle = {Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM)},
  year = {2018},
  month = {June},
  date = {23-28},
  location = {USA}}

```
