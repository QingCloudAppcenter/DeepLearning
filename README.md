# 深度学习简介
2016 年AlphaGo 战胜李世石，预示我们进入了AI时代。深度学习AI的核心技术，在图像分类，自然语言处理，语音识别，无人驾驶等众多领域显示出了强大的能力，各大巨头纷纷投入巨资研发。语音助手，人脸识别，外文翻译等等，AI已融入到了我们生活的方方面面，极大了促进了社会的发展。其中Caffe, TensorFlow, PyTorch 是主流的深度学习框架，拥有强大的社区支持，是实践深度学习的不可或缺的工具。
### Caffe
Caffe 是一个被广泛使用的深度学习框架，由BVLC开发。Cafffe 容易上手，训练速度快，组件模块化，并拥有大量的训练好的经典模型。Caffe 在GPU上训练的性能很好，但只能支持单机多GPU的训练，不支持分布式的训练。
### TensorFlow
TensorFlow 由Google大脑主导开发，是一个异构分布式系统上的大规模深度学习框架。移植性好，可以运行在移动设备上，并支持分布式多机多卡训练，支持多种深度学习模型。TensorFlow 还有功能强大的可视化组件TensorBoard，能可视化网络结构和训练过程，对于观察复杂的网络结构和监控长时间、大规模的训练很有帮助。
### PyTorch
PyTorch 从Torch发展而来，并经过了大量优化，由FaceBook AI 团队主导开发。不同于TensorFlow，PyTorch采用动态计算图的方式，并提供良好的python接口，代码简单灵活。内存分配经过了优化，也能支持分布式训练。
# 青云深度学习平台
青云不仅提供GPU主机（CUDA8.0 + cudnn5），并搭建好了深度学习平台供用户使用。平台上集成了原始的Caffe, TensorFlow, PyTorch, 省去了用户搭建环境的麻烦，提高开发效率。用户无需修改代码，即可把本地的代码运行在云上，并能动态扩展所需资源。
### Caffe 测试示例
Caffe支持python 接口，用户也可以根据需要重新配置编译。目前不支持分布式训练。
####单机
单机示例：  
cd /home/ubuntu/caffe  
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
### TensorFlow 测试示例
TensorFlow 版本号为1.1，支持单机和分布式训练。
#### 单机：
cd /home/ubuntu/tensorflow  
python mnist.py
#### 分布式：
（修改对应的IP地址）  
节点1：  
cd /home/ubuntu/tensorflow  
python mnist_dist.py --ps_hosts=192.168.1.6:2221 --worker_hosts=192.168.1.6:2223,192.168.1.7:2223 --job_name=ps --task_index=0  
python mnist_dist.py --ps_hosts=192.168.1.6:2221 --worker_hosts=192.168.1.6:2223,192.168.1.7:2223 --job_name=worker --task_index=0  
节点2：  
cd /home/ubuntu/tensorflow  
python mnist_dist.py --ps_hosts=192.168.1.6:2221 --worker_hosts=192.168.1.6:2223,192.168.1.7:2223 --job_name=worker --task_index=1

### PyTorch 测试示例
#### 单机 
cd /home/ubuntu/pytorch  
python mnist.py
#### 分布式
节点1：  
cd /home/ubuntu/pytorch   
python mnist_dist.py  
节点2：  
cd /home/ubuntu/pytorch   
python mnist_dist.py
