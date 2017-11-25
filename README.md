# Caffe_vectorizedPooling

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

## School Project [WIP]
This is For School Project. [OpenSourceSoftWare]
### Team Member

###### [ChiHyun Ahn, dept of Computer Information Communication Eng, Hongik Univ](https://github.com/accomplishedboy)

###### [JiHoon Han, dept of Computer Information Communication Eng, Hongik Univ](https://github.com/Hahnnz)
### What we will do
At pooling step, pooling has output Max value that is biggest among activate nodes in kernel or Average value
However, as domain goes, we cannot grant Max pooling or Ave pooling extract good feature to train data

So We Propose Another new way beside Max & Ave pooling

Based on previously existing Max & Average Pooling we will Make Operation Vectorizing With adding Weights


### What we expect from this project
 - <b> Enable to extract MORE effective feature when learn it </b> 
   
   For Each Kernel node, Weights vector can weight heavily if it looks important or get lighter even get 0 to make it unable to join operation if it looks not good or bad  

- <b> It can be learned even if features we want to make it learned are weak relatively in given data </b> 
   
   When we make it learned object on given image, Weight vectors can weight weakly to arounds feature we don’t want to be learned, and weight strongly the feature we want
## How To Use Vectorized Mode?
You can use it by setting 'pool' as "VEC". see below example
```
layer {
         name: "pool_layer"
         type: "Pooling"
         bottom: "convolution_output"
         top: "pool_output"
         pooling_param {
                 pool: VEC
                 kernel_size: 3
                 stride: 2
         }
}
```
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
