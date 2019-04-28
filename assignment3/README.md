# Assignment 3

> 已经省略datasets文件(共2.16G)，可运行`assignment3\cs231n\datasets\get_assignment3_data.sh`下载

[TOC]

## RNN

- `os.remove()`的一个小坑：

  ```python
  def image_from_url(url):
      """
      Read an image from a URL. Returns a numpy array with the pixel data.
      We write the image to a temporary file then read it back. Kinda gross.
      """
      try:
          f = urllib.request.urlopen(url)
          _, fname = tempfile.mkstemp()
          with open(fname, 'wb') as ff:
              ff.write(f.read())
          img = imread(fname)
          
          # os.remove(fname)        
          # 这里只使用os.remove可能会由于文件占用而删除失败，进而导致程序中断而报错
          # 我选择采用多次删除，虽不甚解决，但可一定程度上缓解该问题，即下方注释中的代码
          '''
          count = 0
          while os.path.exists(fname):
          	count += 1
          	if count >= 200:
          		print("can't delete " + fname)
          		break
          	try:
          		os.remove(fname)
          	except:
          		pass
          '''
          
          return img
      except urllib.error.URLError as e:
          print('URL Error: ', e.reason, url)
      except urllib.error.HTTPError as e:
          print('HTTP Error: ', e.code, url)
  ```



- 预设的数据处理统一在`coco_utils.py`，以及两个`h5py`文件中，包含字典建立等等
- RNN反向传播：

  - ![1556444136443](rnn_b.png)

- 在计算loss的时候，由于采用了对变长序列末尾进行填充**\<NULL\>**的操作，因此需要用一个mask罩住，过滤掉填充的部分（即以下代码中的*mask_flat*）

  ```python
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  ```

- RNN 在 test 与 train 不同，首先依旧是根据图片的 feature 映射出初始状态 h(0)，同样，h(0) 作为初始状态，类似于\<SOS\>即*StartOfSentence*，是不参与到计算 caption 的输出的。隐状态 h(1) 的输入是 x(0)，输出 caption 的第一个单词 x(1)，然后以 x(1) 作为第二个隐状态的输入，以此类推直至到序列允许的最大长度结束。

  **需要注意的是，这里序列允许的最大长度 max_length 和训练时一个 time capsule 的时序最大长度 T 没有任何关系**

- **这个初级任务还是蛮有意思的，涉及到很多nlp预处理的内容，很值得继续探索学习**

### word embedding

  - 参考<https://blog.csdn.net/fortilz/article/details/80935136>
  - 充分利用python的index用法，即`out = W[x, :]`和`np.add.at(dW,x,dout)`



## LSTM

- ![1556448653247](lstm_b.png)

- 这里的技巧：
  - **Wx的大小为（D * 4H），可以用切片将其分为四份**
  - 在进行正向传播时，将每个过程所有的cache用字典保存下来
    - 虽然Wh、Wx等参数重复保存了，但由于反向传播需要的中间变量值太多（**i、f、o、g等**）为了方便起见直接用字典保存