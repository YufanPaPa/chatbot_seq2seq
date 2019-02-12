dataset.py	负责把数据存成二进制文件的tfrecords文件。 data/test_span_seq.data是数据例子，你可以根据自己的数据格式修改dataset.py
model.py	模型定义
module.py	model中用到的attention，pointer net等
run.py		程序入口，所有参数配置。
vocabulary.py	词表程序
decode.py	生成程序，其中会调用beamsearch.py
beansearch.py	beamsearch 程序
