<!--
# Model Cards for IQA-PyTorch
 -->
# IQA-PyTorch 的模型卡

<!--
## General FR/NR Methods
 -->
## 通用 FR/NR 方法

<!--
<table>
<tr><td>
 -->
<表> <tr><td>

<!--
| FR Method                | Backward           |
| ------------------------ | ------------------ |
| AHIQ                     | :white_check_mark: |
| PieAPP                   | :white_check_mark: |
| LPIPS                    | :white_check_mark: |
| DISTS                    | :white_check_mark: |
| WaDIQaM                  | :white_check_mark: |
| CKDN<sup>[1](#fn1)</sup> | :white_check_mark: |
| FSIM                     | :white_check_mark: |
| SSIM                     | :white_check_mark: |
| MS-SSIM                  | :white_check_mark: |
| CW-SSIM                  | :white_check_mark: |
| PSNR                     | :white_check_mark: |
| VIF                      | :white_check_mark: |
| GMSD                     | :white_check_mark: |
| NLPD                     | :white_check_mark: |
| VSI                      | :white_check_mark: |
| MAD                      | :white_check_mark: |
 -->
| FR法|向后| | ------------------------ | ------------------ | | AHIQ | :white_check_mark: | |派APP | :white_check_mark: | | LPIPS | :white_check_mark: | |距离 | :white_check_mark: | |瓦迪卡姆 | :white_check_mark: | | CKDN<sup>[1](#fn1)</sup> | :white_check_mark: | | FSIM| :white_check_mark: | | SSIM | :white_check_mark: | | MS-SSIM | :white_check_mark: | | CW-SSIM | :white_check_mark: | |峰值信噪比 | :white_check_mark: | | VIF| :white_check_mark: | | GMSD | :white_check_mark: | |国家警察局 | :white_check_mark: | | VSI| :white_check_mark: | |疯狂 | :white_check_mark: |

<!--
</td><td>
 -->
</td><td>

<!--
| NR Method                    | Backward                 |
| ---------------------------- | ------------------------ |
| FID                          | :heavy_multiplication_x: |
| CLIPIQA(+)                   | :white_check_mark:       |
| MANIQA                       | :white_check_mark:       |
| MUSIQ                        | :white_check_mark:       |
| DBCNN                        | :white_check_mark:       |
| PaQ-2-PiQ                    | :white_check_mark:       |
| HyperIQA                     | :white_check_mark:       |
| NIMA                         | :white_check_mark:       |
| WaDIQaM                      | :white_check_mark:       |
| CNNIQA                       | :white_check_mark:       |
| NRQM(Ma)<sup>[2](#fn2)</sup> | :heavy_multiplication_x: |
| PI(Perceptual Index)         | :heavy_multiplication_x: |
| BRISQUE                      | :white_check_mark:       |
| ILNIQE                       | :white_check_mark:       |
| NIQE                         | :white_check_mark:       |
</tr>
</table>
 -->
| NR法|向后| | ---------------------------- | ------------------------ | | FID | :heavy_multiplication_x: | | CLIPIQA(+) | :white_check_mark: | |马尼卡| :white_check_mark: | |音乐 | :white_check_mark: | |数据库CNN | :white_check_mark: | | PaQ-2-PiQ | :white_check_mark: | |超级IQA | :white_check_mark: | |尼玛| :white_check_mark: | |瓦迪卡姆 | :white_check_mark: | | CNNIQA | :white_check_mark: | | NRQM(马)<sup>[2](#fn2)</sup> | :heavy_multiplication_x: | | PI（感知指数）| :heavy_multiplication_x: | |清爽 | :white_check_mark: | |伊尔尼克 | :white_check_mark: | |尼科 | :white_check_mark: | </tr> </表>

<!--
<a name="fn1">[1]</a> This method use distorted image as reference. Please refer to the paper for details.<br>
<a name="fn2">[2]</a> Currently, only naive random forest regression is implemented and **does not** support backward.
 -->
<a name="fn1">[1]</a> 该方法使用扭曲图像作为参考。详情请参阅论文。<br><a name="fn2">[2]</a>目前仅实现了朴素随机森林回归，**不**支持向后。

<!--
## IQA Methods for Specific Tasks
 -->
## 针对特定任务的 IQA 方法

<!--
| Task           | Method  | Description                                                                                                                                                                 |
| -------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Underwater IQA | URanker | A ranking-based underwater image quality assessment (UIQA) method, AAAI2023, [Arxiv](https://arxiv.org/abs/2208.06857), [Github](https://github.com/RQ-Wu/UnderwaterRanker) |
 -->
|任务|方法|描述 | | -------------- | -------- | -------------------------------------------------- -------------------------------------------------- -------------------------------------------------- -------------------- | |水下IQA | URanker |基于排名的水下图像质量评估（UIQA）方法，AAAI2023，[Arxiv](https://arxiv.org/abs/2208.06857)，[Github](https://github.com/RQ-Wu/UnderwaterRanker) |

<!--
## Outputs of Different Metrics 
**Note: `~` means that the corresponding numeric bound is typical value and not mathematically guaranteed**
 -->
## 不同指标的输出 **注意：`~` 表示对应的数值范围是典型值，并且没有数学保证**

<!--
| model    | lower better ? | min | max     | DATE | Link                                                                                                                                                      |
| -------- | -------------- | --- | ------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| clipiqa  | False          | 0   | 1       | 2022 | https://arxiv.org/abs/2207.12396                                                                                                                          |
| maniqa   | False          | 0   |        | 2022 | https://arxiv.org/abs/2204.08958                                                                                                                          |
| hyperiqa | False          | 0   | 1       | 2020 | [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf)                 |
| cnniqa   | False          |   |       | 2014 | [pdf](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)                                      |
| tres     | False          |    | | 2022 | https://github.com/isalirezag/TReS                                                                                                                        |
| musiq    | False          |  ~0 | ~100 | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| musiq-ava    | False          |  ~0  | ~10 | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| musiq-koniq    | False          | ~0 | ~100 | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| musiq    | False          |    | | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| paq2piq  | False          |    | | 2020 | [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ying_From_Patches_to_Pictures_PaQ-2-PiQ_Mapping_the_Perceptual_Space_of_CVPR_2020_paper.pdf) |
| dbcnn    | False          |    | | 2019 | https://arxiv.org/bas/1907.02665                                                                                                                          |
| brisque  | True           |    | | 2012 | [pdf](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf)                                                                                    |
| pi       | True           |    | | 2018 | https://arxiv.org/abs/1809.07517                                                                                                                          |
| nima     | False          |   | | 2018 | https://arxiv.org/abs/1709.05424                                                                                                                          |
| nrqm     | False          |   | | 2016 | https://arxiv.org/abs/1612.05890                                                                                                                          |
| ilniqe   | True           | 0   | | 2015 | [pdf](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IL-NIQE.pdf)                                                                                          |
| niqe     | True           | 0   | | 2012 | [pdf](https://live.ece.utexas.edu/publications/2013/mittal2013.pdf)                                                                                       | -->
|型号|越低越好？ |分钟 |最大|日期 |链接 | | -------- | -------------- | --- | -------- | ---- | -------------------------------------------------- -------------------------------------------------- -------------------------------------------------- --- | |剪辑|假 | 0 | 1 | 2022 | 2022 https://arxiv.org/abs/2207.12396 | |马尼卡 |假 | 0 | | 2022 | 2022 https://arxiv.org/abs/2204.08958 | |超级iqa |假 | 0 | 1 | 2020 | [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) | | cnniqa |假 | | | 2014年| [pdf](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) | |特雷斯|假 | | | 2022 | 2022 https://github.com/isalirezag/TReS | |音乐|假 | 〜0 | 〜100 | 2021 | https://arxiv.org/abs/2108.05997 | |音乐-ava |假| 〜0 | 〜10 | 2021 | https://arxiv.org/abs/2108.05997 | |音乐-koniq |假 | 〜0 | 〜100 | 2021 | https://arxiv.org/abs/2108.05997 | |音乐|假| | | 2021 | https://arxiv.org/abs/2108.05997 | | paq2piq |假 | | | 2020 | [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ying_From_Patches_to_Pictures_PaQ-2-PiQ_Mapping_the_Perceptual_Space_of_CVPR_2020_paper.pdf) | |数据库网络 |假 | | | 2019 | 2019 https://arxiv.org/bas/1907.02665 | |布里斯克|真实| | | 2012 | [pdf](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) | |圆周率 |真实| | | 2018 | https://arxiv.org/abs/1809.07517 | |尼玛|假 | | | 2018 | https://arxiv.org/abs/1709.05424 | | NRQM |假 | | | 2016 | 2016 https://arxiv.org/abs/1612.05890 | |伊尔尼克 |真实| 0 | | 2015 | 2015 [pdf](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IL-NIQE.pdf) | |尼奇 |真实| 0 | | 2012 | [pdf](https://live.ece.utexas.edu/publications/2013/mittal2013.pdf) |

