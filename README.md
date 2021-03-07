# mgru_gate_re
mogrifier bigru-gateattention-softmax有监督关系抽取的代码

# 文件说明
* /data文件夹：存放词嵌入文件与关系抽取源文件
* /experiments文件夹：存放各模型参数及实验记录
* model文件夹：存放模型文件net.py，其中包括CNN模型、Att-BLSTM模型、Att-Transformer模型、Mogrifier BiGRU-GateAttention-softmax模型
* plot文件夹：实验记录绘图
* build_semeval_dataset.py：关系抽取文件预处理
* build_vocab.py:在build_semeval_dataset.py处理的基础上进一步处理
* train.py：模型主函数
