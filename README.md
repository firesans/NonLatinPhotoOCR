# Convolutional Recurrent Neural Network + CTCLoss | STAR-Net

Code for paper "[Towards Boosting the Accuracy of Non-Latin Scene Text Recognition](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2021/Improving_Arabic_STR_accuracies__ASAR21.pdf)"

## Dependence

- Python3.6.5
- torch==1.2.0
- torchvision==0.4.0
- tensorboard==2.3.0

## How to run the code?

### Prepare data

- Follow the instructions in [meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>) to create lmdb datasets.
	Use the same step to create train and val data.

### Change parameters and alphabets

Please update the parameters and alphabets according to the requirement. 

- Change parameters in the mytrain.py file
- Change alphabets

  Please put all the alphabets that appear in your labels in a file and input the list as charlist to mytrain.py, else the program will throw an error during training.

### Train

Run `mytrain.py` - 

```sh
python3 mytrain.py --trainRoot /ssd_scratch/cvit/sanjana/hindi-train-lmdb \
--valRoot /ssd_scratch/cvit/sanjana/hindi-test-lmdb \
--arch crnn --lan hindi --charlist /ssd_scratch/cvit/sanjana/crnn_new/lexicon.txt \
--batchSize 32 --nepoch 15 --cuda --expr_dir /ssd_scratch/cvit/sanjana \
--displayInterval 10 --valInterval 100 --adadelta \ 
--manualSeed 1234 --random_sample --deal_with_lossnan 
```

## Reference

[meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>) \
[Sierkinhane/crnn_chinese_characters_rec](<https://github.com/Sierkinhane/crnn_chinese_characters_rec>)

If you use the dataset or code from this work, please add the following citation:-

```
@inproceedings{gunnaNonLatin2021,
  title={Towards {B}oosting the {A}ccuracy of {N}on-{L}atin {S}cene {T}ext {R}ecognition,
  author={Sanjana Gunna and Rohit Saluja and C V Jawahar},
  booktitle={2021 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2021},
  organization={IEEE}
}
```
