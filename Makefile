cc        := g++
nvcc      = ${cuda_home}/bin/nvcc


cuda_home := /usr/local/anaconda3/lib/python3.8/site-packages/trtpy/trt8cuda115cudnn8
syslib    := /usr/local/anaconda3/lib/python3.8/site-packages/trtpy/lib
cpp_pkg   := /usr/local/anaconda3/lib/python3.8/site-packages/trtpy/cpp-packages

use_python     := true
python_root    := /usr/local/anaconda3


python_name    := python3.8

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
cuda_arch := # -gencode=arch=compute_75,code=sm_75

cpp_srcs  := $(shell find src -name "*.cpp")
cpp_objs  := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs  := $(cpp_objs:src/%=objs/%)
cpp_mk    := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs  := $(shell find src -name "*.cu")
cu_objs  := $(cu_srcs:.cu=.cu.o)
cu_objs  := $(cu_objs:src/%=objs/%)
cu_mk    := $(cu_objs:.cu.o=.cu.mk)

include_paths := src        \
			src/application \
			src/tensorRT	\
			src/tensorRT/common  \
			$(cuda_home)/include/cuda     \
			$(cuda_home)/include/tensorRT \
			$(cuda_home)/include/cudnn \
			$(cpp_pkg)/opencv-4.2.0/include/opencv4  \
			$(cuda_home)/include/protobuf

library_paths := $(cuda_home)/lib64 $(syslib) $(cpp_pkg)/opencv-4.2.0/lib /usr/lib/x86_64-linux-gnu

link_librarys := tiff opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvinfer_plugin \
			cuda cublas cudart cudnn \
			stdc++ protobuf dl


# HAS_PYTHON表示是否编译python支持
support_define    := 

ifeq ($(use_python), true) 
include_paths  += $(python_root)/include/$(python_name)
library_paths  += $(python_root)/lib
link_librarys  += $(python_name)
support_define += -DHAS_PYTHON
endif

empty         :=
export_path   := $(subst $(empty) $(empty),:,$(library_paths))

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=c++11 -g -w -O0 -fPIC -pthread -fopenmp $(support_define)
cu_compile_flags  := -std=c++11 -g -w -O0 -Xcompiler "$(cpp_compile_flags)" $(cuda_arch) $(support_define)
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

pytrtc : example-python/pytrt/libpytrtc.so
expath : library_path.txt

library_path.txt : 
	@echo LD_LIBRARY_PATH=$(export_path):"$$"LD_LIBRARY_PATH > $@



example-python/pytrt/libpytrtc.so : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

objs/%.cpp.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

objs/%.cu.o : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

objs/%.cpp.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
	
objs/%.cu.mk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)


workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -g -o $@ $(link_flags)

self_driving : workspace/pro
	@cd workspace && ./pro self_driving

clean :
	@rm -rf objs workspace/pro example-python/pytrt/libpytrtc.so example-python/build example-python/dist example-python/pytrt.egg-info example-python/pytrt/__pycache__
	@rm -rf build
	@rm -rf example-python/pytrt/libplugin_list.so
	@rm -rf library_path.txt

.PHONY : clean self_driving

# 导出符号，使得运行时能够链接上
export LD_LIBRARY_PATH:=$(export_path):$(LD_LIBRARY_PATH)