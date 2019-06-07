# textsimilarity_based_bert

version_1.py
将两句组合成cls+句1+sep+句2+sep输入bert，取cls位置输出进全连接层判断相似性。

version_2.py
将分别获取两句话的bert输出句向量，再对两个句向量交互，类似Siamese。


以上两种方法在训练集上都不拟合，心态爆炸TT
