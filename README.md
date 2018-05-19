# Forked from weinman/cnn_lstm_ctc_ocr for the ICPR MTWI 2018 challenge 1

### Introduction
This is a repository forked from [weinman/cnn_lstm_ctc_ocr](https://github.com/weinman/cnn_lstm_ctc_ocr) for the [ICPR MTWI 2018 challenge1](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11409391.333.4.7cb749ecJ29ZG7&raceId=231650).
<br>Origin Repository: [weinman/cnn_lstm_ctc_ocr - A tensorflow implementation of end to end text recognition](https://github.com/weinman/cnn_lstm_ctc_ocr)
<br>Origin Author: [weinman](https://github.com/weinman)

Author: [Feng zhang](https://github.com/zfxxfeng)
<br>Email: 364051598@qq.com

### Contents
1. [Transform](#transform)
2. [Models](#models)
3. [Demo](#demo)
3. [Train](#train)
4. [Test](#test)
5. [Results](#results)

### Transform
You should cut the data by yourself.Use the target_cut.py. Change the data path to your own origin data.I use the dataset like [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.33e4b86aZXVkts&raceId=231650). Note,I use Affine changes the change the Oblique picture to rectangle.So I need the clockwise direction of target label. Use the getTxt.py to change the label's direction.

Next, use mjsynth-tfrecord.py to change your data into tfrecord.You can Find the way in [[weinman/cnn_lstm_ctc_ocr/Makefile](https://github.com/weinman/cnn_lstm_ctc_ocr/blob/master/Makefile)] You only need to change some paths.

### Models
I use the new word_dictionary which consists of English, Chinese and number.I only upload a old pretrain model,May it works badly. If you train it for one day with your data,it will work well.And,I add some data augmentation for the model.
You also can change the model to denseNet,it will work better.Do as follows in the train.py :
'''
Import denseNet 
%features,sequence_length = model.convnet_layers( image, width, mode)
features,sequence_length = zf_mod_denseNet2.Dense_net( image, width, mode)
'''
Models trained on [ICPR MTWI 2018 (train)](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.33e4b86aZXVkts&raceId=231650): [[model_download](https://pan.baidu.com/s/15IxbqsiuxFyAx8zFsCVe8g)].The password is ydtv.
Some English data can find in [[weinman](http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz)]
### Demo
Download models and copy it to data/model.Run:
```
python validate.py  your picture's path
eg.python validate.py E:/1.jpg
```


### Train
When we make your data to tfrecord,You can train. 
```
cd src
python train.py
```


### Test
use also can test the tfrecord's accuracy.[[usage](https://github.com/weinman/cnn_lstm_ctc_ocr#testing)]
put your data in src/data/val and do as follows:
```
cd src;
python test.py
```

### Results
Here are some results on [ICPR MTWI 2018](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.5022b86af5JwV4&raceId=231650):
<br>![image](https://github.com/zfxxfeng/cnn_lstm_ctc_ocr_for_ICPR/blob/master/data/test_image/result.jpg)
# Enjoy yourself



