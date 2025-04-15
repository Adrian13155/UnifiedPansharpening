# Unified Pansharpening

### 代码说明

* 本代码`fork`自https://github.com/TingMAC/DURE-Net，绝大部分的深度展开框架的代码都来自于这个仓库，但是也是可以用的。
* 深度展开框架中涉及到了两个ProxNet，一个是用了PromptIR，是对于每张图像的特有的先验；另一个则是用了codebook，是对于数据集自身的先验。
* PromptIR的代码大部分来自https://github.com/va1shn9v/PromptIR/blob/main/net/model.py，由于源代码有很多耦合度太高的地方，所以做了一点修改。
  * 对应的代码在`PromptIR/Model_AMIR.py`中的*ProxNet_Prompt*类
* Codebook中`codebook/model/model3D`目录下的代码以及`codebook/model/vq3D.py`都是改成3D的代码，可以用来参考，其中`codebook/model/model3D/MSAB3D.py`中涉及到一些注意力的代码改成3D的，会比较麻烦，可以参考借鉴。

#### Dataset

`Dataset.py`中的`MatWithTextDataset`是加载遥感图像数据集和文本`Caption`数据集的代码。

### 深度展开框架

* `Model.py`是中的`DURE`是2D深度展开框架的代码。

* `Model2D_3D.py`中的`DURE2D_3D`是既有2D又有3D的框架的代码，除了codebook是3D的，其他的部分全都是2D的。

> 因为发现纯3D框架效果比较差，所以看一下把除codebook外的代码都改成2D看看是什么效果/

* `Model3D.py`中的`DURE3D`是3D深度展开框架的代码，其中的`self.proxNet`目前是有3种选择，都写在类的注释下了。

> 其中的WavBest是论文Empower Generalizability for Pansharpening Through Text-Modulated Diffusion Model中的3D-Unet魔改过来的。
>
> ProxNet_Prompt3D_WithTextPrompt是将InstructIR中加文本提示的方式改成了3D的后搬过来的。



### 训练

`train_codebook.py`是训练`codebook`的代码。

`train_pansharpening.py`是训练深度展开框架的代码。

### 理论

![image-20250328163124178](Unified Pansharpening.assets/Unified Pansharpening.png)

$F$是HRMS，$D$是模糊核，$M$是LRMS。$H$是光谱变换(也许)，$P$是PAN图像。

其中$R_s$和$R_t$就是分别指的对于数据集的先验和对于每个图像的先验。