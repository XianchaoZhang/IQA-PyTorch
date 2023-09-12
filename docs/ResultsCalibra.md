<!--
# Results Calibration
 -->
# 结果校准

<!--
We random select 5 pairs of images from TID2013 for results calibration. Images are stored under `./dist_dir` and `./ref_dir`. Results of different metrics are saved under `./results_compare/`. We also record the problems encountered during our reproduction of matlab scripts in [MatlabReproduceNote](./MatlabReproduceNote.md)
 -->
我们从 TID2013 中随机选择 5 对图像进行结果校准。图像存储在 `./dist_dir` 和 `./ref_dir` 下。不同指标的结果保存在 `./results_compare/` 下。我们还将我们在复制 matlab 脚本过程中遇到的问题记录在 [MatlabReproductNote](./MatlabReproductNote.md) 中


| Method                                              | I03.bmp                   | I04.bmp          | I06.bmp | I08.bmp | I19.bmp  | Speed (/image) |
| --------------------------------------------------- | ------------------------- | ---------------- | ------- | ------- | -------- | -------------- |
| CKDN<sup>[1](#fn1)</sup>(org)                       | 0.2833                    | 0.5766           | 0.6367  | 0.6579  | 0.5999   |
| CKDN(ours imported)                                 | 0.2833                    | 0.5766           | 0.6367  | 0.6579  | 0.5999   |
| LPIPS(org)                                          | 0.7237                    | 0.2572           | 0.0508  | 0.0521  | 0.4253   |
| LPIPS(ours imported)                                | 0.7237                    | 0.2572           | 0.0508  | 0.0521  | 0.4253   |
| DISTS(org)                                          | 0.4742                    | 0.1424           | 0.0682  | 0.0287  | 0.3123   |
| DISTS(ours imported)                                | 0.4742                    | 0.1424           | 0.0682  | 0.0287  | 0.3123   |
| SSIM<sup>[2](#fn2)</sup>(org)                       | 0.6993                    | 0.9978           | 0.9989  | 0.9669  | 0.6519   |
| SSIM(ours imported)                                 | 0.6997                    | 0.9978           | 0.9989  | 0.9671  | 0.6521   |
| MS-SSIM<sup>[3](#fn3)</sup>(org)                    | 0.6733                    | 0.9996           | 0.9998  | 0.9566  | 0.8462   |
| MS-SSIM(ours imported)                              | 0.6698                    | 0.9993           | 0.9996  | 0.9567  | 0.8418   |
| CW-SSIM<sup>[9](#fn9)</sup>(org)                    | 0.2763                    | 0.9996           | 1.0000  | 0.9068  | 0.8658   |
| CW-SSIM(ours imported)                              | 0.2782                    | 0.9995           | 1.0000  | 0.9065  | 0.8646   |
| PSNR<sup>[4](#fn4)</sup>(org)                       | 21.11                     | 20.99            | 27.01   | 23.30   | 21.62    |
| PSNR(ours imported)                                 | 21.11                     | 20.99            | 27.01   | 23.30   | 21.62    |
| FSIM(org)                                           | 0.6890                    | 0.9702           | 0.9927  | 0.9575  | 0.8220   |
| FSIM(ours imported)                                 | 0.6891                    | 0.9702           | 0.9927  | 0.9575  | 0.8220   |
| VIF<sup>[5](#fn5)</sup>(org)                        | 0.0172                    | 0.9891           | 0.9924  | 0.9103  | 0.1745   |
| VIF(ours imported)                                  | 0.0172                    | 0.9891           | 0.9924  | 0.9103  | 0.1745   |
| GMSD<sup>[6](#fn6)</sup>(org)                       | 0.2203                    | 0.0005           | 0.0004  | 0.1346  | 0.2050   |
| GMSD(ours imported)                                 | 0.2203                    | 0.0005           | 0.0004  | 0.1346  | 0.2050   |
| NLPD<sup>[7](#fn7)</sup>(org)                       | 0.5616                    | 0.0195           | 0.0159  | 0.3028  | 0.4326   |
| NLPD(ours imported)                                 | 0.5616                    | 0.0139           | 0.0110  | 0.3033  | 0.4335   |
| VSI<sup>[8](#fn8)</sup>(opt)                        | 0.9139                    | 0.9620           | 0.9922  | 0.9571  | 0.9262   |
| VSI(ours imported)                                  | 0.9244                    | 0.9497           | 0.9877  | 0.9541  | 0.9348   |
| MAD<sup>[10](#fn10)</sup>(ours imported)            | 194.9324                  | 0.0000           | 0.0000  | 91.6206 | 181.9651 |
| NIQE<sup>[11](#fn11)</sup>(org)                     | 15.7536                   | 3.6549           | 3.2355  | 3.1840  | 8.6352   |
| NIQE(ours imported)                                 | 15.6530                   | 3.6541           | 3.2343  | 3.2076  | 9.1060   |
| ILNIQE(org)                                         | 113.4801                  | 23.9968          | 19.9750 | 22.4493 | 56.6721  | 10s            |
| ILNIQE(ours imported)                               | 115.6144                  | 24.0634          | 19.7497 | 22.3253 | 54.7657  | 1s             |
| BRISQUE<sup>[12](#fn12)</sup>(org)                  | 94.6421                   | -0.1076          | 0.9929  | 5.3583  | 72.2617  |
| BRISQUE(ours imported)                              | 94.6448                   | -0.1103          | 1.0772  | 5.1418  | 66.8405  |
| MUSIQ/AVA(org)                                      | 3.398                     | 5.648            | 4.635   | 5.186   | 4.128    |
| MUSIQ/AVA(ours imported)(org)<sup>[13](#fn13)</sup> | 3.408                     | 5.693            | 4.696   | 5.196   | 4.195    |
| MUSIQ/koniq10k(org)                                 | 12.494                    | 75.332           | 73.429  | 75.188  | 36.938   |
| MUSIQ/koniq10k(ours imported)                       | 12.477                    | 75.776           | 73.745  | 75.460  | 38.02    |
| MUSIQ/paq2piq(org)                                  | 46.035                    | 72.660           | 73.625  | 74.361  | 69.006   |
| MUSIQ/paq2piq(ours imported)                        | 46.018                    | 72.665           | 73.765  | 74.387  | 69.721   |
| MUSIQ/spaq(org)                                     | 17.685                    | 70.492           | 78.740  | 79.015  | 49.105   |
| MUSIQ/spaq(ours imported)                           | 17.680                    | 70.653           | 79.036  | 79.318  | 50.452   |
| NRQM                                                | 1.3894                    | 8.9394           | 8.9735  | 6.8290  | 6.3120   | 10s            |
| NRQM (ours imported)                                | 1.3931 | 8.9418 | 8.9721 | 6.8309 | 6.3031 | 5s             |
| PI<sup>[14](#fn14)</sup>                            | 11.9235                   | 3.0720           | 2.6180  | 2.8074  | 6.7713   |
| PI (ours imported )                                 | 11.9286 | 3.0730 | 2.6356 | 2.7979 | 6.9545   |
| Paq2piq                                             | 44.1340                   | 73.6015          | 74.3297 | 76.8748 | 70.9153  |
| Paq2piq (ours imported)                             | 44.1340                   | 73.6015          | 74.3297 | 76.8748 | 70.9153  |
| PieAPP                                              | 4.2976                    | 3.9088           | 2.2620  | 1.4274  | 3.4188   |
| PieAPP (ours imported)                              | 4.2976                    | 3.9088           | 2.2620  | 1.4274  | 3.4188   |
| FID<sup>[15](#fn15)</sup>                           | 225.3678 (legacy_pytorch) | 220.5819 (clean) |         |         |          |
| FID (ours imported)                                 | 225.3679 (legacy_pytorch) | 220.5819 (clean) |         |         |          |



<!--
#### Notice
<a name="fn1">[1]</a> CKDN used degraded images as references in the original paper.<br>
<a name="fn2">[2]</a> The original SSIM matlab script downsample the image when larger than 256. We remove such constraint. We use rgb2gray function as input of original SSIM matlab script<br>
<a name="fn3">[3]</a> We use rgb2gray function as input of original MS-SSIM matlab script.<br>
<a name="fn4">[4]</a> The original PSNR code refers to scikit-learn package with RGB 3-channel calculation (from skimage.metrics import peak_signal_noise_ratio).<br>
<a name="fn5">[5]</a> We use rgb2gray function as input of original VIF matlab script.<br>
<a name="fn6">[6]</a> We use rgb2gray function as input of original GMSD matlab script.<br>
<a name="fn7">[7]</a> We use rgb2gray function as input of original NLPD matlab script, and try to mimic 'imfilter' and 'conv2' functions in matlab.<br>
<a name="fn8">[8]</a> Since official matlab code is not available, we use the implement of IQA-Optimization for comparation. The differences are described as follows. After modifying the above implementation, the results are basically the same.
 -->
#### 注意
<a name="fn1">[1]</a> CKDN 在原始论文中使用了降级图像作为参考。<br>
<a name="fn2">[2]</a>原始 SSIM matlab 脚本在大于 256 时对图像进行下采样。我们删除了这样的约束。我们使用 rgb2gray 函数作为原始 SSIM matlab 脚本的输入<br>
<a name="fn3">[3]</a> 我们使用 rgb2gray 函数作为原始 MS-SSIM matlab 脚本的输入。<br>
<a name= "fn4">[4]</a> 原始 PSNR 代码参考 scikit-learn 包进行 RGB 3 通道计算（from skimage.metrics importpeak_signal_noise_ratio）。<br>
<a name="fn5">[5] </a> 我们使用 rgb2gray 函数作为原始 VIF matlab 脚本的输入。<br> <a name="fn6">[6]</a> 我们使用 rgb2gray 函数作为原始 GMSD matlab 脚本的输入。<br>
< a name="fn7">[7]</a> 我们使用 rgb2gray 函数作为原始 NLPD matlab 脚本的输入，并尝试模仿 matlab 中的 'imfilter' 和 'conv2' 函数。<br>
<a name="fn8 ">[8]</a> 由于没有官方的 matlab 代码，我们使用 IQA-Optimization 的实现来进行比较。差异描述如下。修改上述实现后，结果基本相同。

<!--
- we use interpolation to transform the image to 256*256 and then back to the image size after calculating VSMap in the SDSP function
- rgb2lab's function is slightly different
- the range of ours is -127 to 128 when constructing SDMap, and the value of optimization is -128 to 127
- different down-sampling operations
 -->
- 我们使用插值将图像转换为 256*256，然后在 SDSP 函数中计算 VSMap 后返回到图像大小
- rgb2lab 的功能略有不同
- 我们构建 SDMap 时的范围是 -127 到 128，优化的值为 -128 到 127
- 不同的下采样操作

<!--
<a name="fn9">[9]</a> We use rgb2gray function as input of original CW-SSIM matlab script. The number of level is 4 and orientation is 8.<br>
<a name="fn10">[10]</a> We use rgb2yiq function as input, and the original MAD matlab script is not available.<br>
<a name="fn11">[11]</a> We use rgb2gray function as input of original NIQE matlab script.<br>
<a name="fn12">[12]</a> We use rgb2gray function images as input of original BRISQUE matlab script.<br>
<a name="fn13">[13]</a> Results have about ±2% difference with tensorflow codes because of some detailed implementation differences between TensorFlow and PyTorch. For example, PyTorch does not support gaussian interpolation, different default epsilon value, etc.<br>
<a name="fn14">[14]</a> Perceptual Index (PI) use YCBCR color space and crop border with size 4.<br>
<a name="fn15">[15]</a> We use codes from the [clean-fid](https://github.com/GaParmar/clean-fid) project.<br>
 -->
<a name="fn9">[9]</a> 我们使用 rgb2gray 函数作为原始 CW-SSIM matlab 脚本的输入。层数为 4，方​​向为 8。<br>
<a name="fn10">[10]</a> 我们使用 rgb2yiq 函数作为输入，原始 MAD matlab 脚本不可用。<br>
< a name="fn11">[11]</a> 我们使用 rgb2gray 函数作为原始 NIQE matlab 脚本的输入。<br>
<a name="fn12">[12]</a> 我们使用 rgb2gray 函数图像作为输入原始 BRISQUE matlab 脚本。<br>
<a name="fn13">[13]</a> 由于 TensorFlow 和 PyTorch 之间存在一些详细的实现差异，结果与张量流代码大约有 ±2% 的差异。例如，PyTorch 不支持高斯插值、不同的默认 epsilon 值等。<br>
<a name="fn14">[14]</a> 感知指数 (PI) 使用 YCBCR 颜色空间和大小为 4 的裁剪边框.<br> <a name="fn15">[15]</a> 我们使用 [clean-fid](https://github.com/GaParmar/clean-fid) 项目中的代码。<br>
