<!--
# Matlab Reproduce Note
 -->
# Matlab 重现笔记

<!--
We record problems encountered during our reproduction of matlab based metrics here for your reference.
 -->
我们在这里记录一下我们在基于 matlab 的指标复现过程中遇到的问题，供大家参考。

<!--
## The function [conv2](https://ww2.mathworks.cn/help/matlab/ref/conv2.html?lang=en) in Matlab
 -->
## Matlab 中的函数 [conv2](https://ww2.mathworks.cn/help/matlab/ref/conv2.html?lang=en)

<!--
- The convolution kernel first rotate by 180 degrees and then compute convolutional results. This problem can be solved with `rotate` or `flip' in Pytorch.
 -->
- 卷积核首先旋转 180 度，然后计算卷积结果。这个问题可以通过 Pytorch 中的 'rotate' 或 'flip' 来解决。

<!--
## The function [imfilter](https://ww2.mathworks.cn/help/images/ref/imfilter.html?lang=en) in Matlab
 -->
## Matlab 中的函数 [imfilter](https://ww2.mathworks.cn/help/images/ref/imfilter.html?lang=en)

<!--
- The Padding Option `symmetric` use mirror reflection of its boundary to pad containing the outermost boundary of this image.
 -->
- 填充选项 `symmetric` 使用其边界的镜面反射来填充包含该图像的最外层边界。
<!--
- The default Padding Option is zero-padding.
 -->
- 默认填充选项是零填充。
<!--
- The default Correlation and Convolution Option is `corr`, which calculate convolutions without rotate.
 -->
- 默认的相关和卷积选项是 `corr`，它计算不旋转的卷积。

