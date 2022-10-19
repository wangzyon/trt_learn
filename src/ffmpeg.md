github 下载 ffmpeg

```
# 将ffmpeg的lib和include就安装到其所在目录，而非自动到系统的lib
./configure --prefix=/usr/local/anaconda3/lib/python3.8/site-packages/trtpy/cpp-packages/ffmpeg-3.0
```

github 下载 nasm，yasm，也可以直接 pip install
源码安装需要将 bin 路径添加到 bashrc 的 PATH

github 下载 x264

```
# 没有configure使用./autogen.sh生成;
# make错误可能需要修改一些文件权限；
# x264直接make && make install到ffmpeg的目录去，即生成到ffmpeg的lib和include

./configure --prefix=/usr/local/anaconda3/lib/python3.8/site-packages/trtpy/cpp-packages/ffmpeg-3.0 --enable-shared
```
