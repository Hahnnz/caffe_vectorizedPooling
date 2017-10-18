# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

## School Project [WIP]
This is For School Project. [OpenSourceSoftWare]
### Team Member

###### [ChiHyeon Ahn, dept of Computer Information Communication, Hongik Univ.](https://github.com/accomplishedboy)

###### [JiHoon Han, dept of Computer Information Communication, Hongik Univ.](https://github.com/Hahnnz)
### What we will do
At pooling step, pooling has output Max value that is biggest among activate nodes in kernel or Average value
However, as domain goes, we cannot grant Max pooling or Ave pooling extract good feature to train data

So We Propose Another new way beside Max & Ave pooling

Based on previously existing Max & Average Pooling we will Make Operation Vectorizing With adding Weights


### What we expect from this project
- Enable to extract MORE effective feature when learn it 

   For Each Kernel node, Weights vector can weight heavily if it looks important or get lighter even get 0 to make it unable to join operation if it looks not good or bad  

- It can be learned even if features we want to make it learned are weak relatively in given data 

   When we make it learned object on given image, Weight vectors can weight weakly to arounds feature we don’t want to be learned, and weight strongly the feature we want
## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
