#include "lane.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <tensorRT/infer/trt_infer.hpp>
#include <tensorRT/common/ilogger.hpp>
#include <tensorRT/common/infer_controller.hpp>
#include <tensorRT/common/preprocess_kernel.cuh>
#include <tensorRT/common/monopoly_allocator.hpp>
#include <tensorRT/common/cuda_tools.hpp>
#include <iostream>

namespace Lane
{
    using namespace cv;
    using namespace std;

    void decode_to_depth_invoker(const float *input, unsigned char *output, int edge, cudaStream_t stream);

    using ControllerImpl = InferController<
        Mat,               // input
        points,            // output
        tuple<string, int> // start param
        >;
    class InferImpl : public Infer, public ControllerImpl
    {
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl()
        {
            stop();
        }

        virtual bool startup(
            const string &file, int gpuid)
        {
            float mean[] = {0.0, 0.0, 0.0};
            float std[] = {1.0, 1.0, 1.0};
            normalize_ = CUDAKernel::Norm::mean_std(
                mean, std, 1 / 255.0f, CUDAKernel::ChannelType::Invert);
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        virtual void worker(promise<bool> &result) override
        {

            string file = get<0>(start_param_);
            int gpuid = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if (engine == nullptr)
            {
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            int max_batch_size = engine->get_max_batch_size();
            auto input = engine->tensor("input.1");
            auto output = engine->tensor("output");

            input_width_ = input->size(3);
            input_height_ = input->size(2);

            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_ = engine->get_stream();
            gpu_ = gpuid;
            result.set_value(true);
            input->resize_single_dim(0, max_batch_size).to_gpu();

            float row_anchors[18] = {121.0, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287};
            vector<Job> fetch_jobs;
            while (get_jobs_and_wait(fetch_jobs, max_batch_size))
            {

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    auto &job = fetch_jobs[ibatch];
                    auto &mono = job.mono_tensor->data();

                    if (mono->get_stream() != stream_)
                    {
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {

                    auto &job = fetch_jobs[ibatch];
                    float input_width_stride = job.input.cols / 800.0f;
                    float input_height_stride = job.input.rows / 288.0f;

                    job.output.resize(18 * 4);
                    int cur = 0;
                    for (int i = 0; i < 72; i++)
                    {
                        auto &point = job.output[cur];
                        float *pitem = output->cpu<float>(ibatch, i);
                        if (*(pitem + 1) == 200.0)
                            continue;
                        point.x = *pitem * input_width_stride;
                        point.y = row_anchors[i / 4] * input_height_stride;
                        cur++;
                    }
                    job.output.resize(cur);
                    job.pro->set_value(job.output);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job &job, const Mat &image) override
        {

            if (tensor_allocator_ == nullptr)
            {
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            if (image.empty())
            {
                INFOE("Image is empty");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if (job.mono_tensor == nullptr)
            {
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto &tensor = job.mono_tensor->data();
            TRT::CUStream preprocess_stream = nullptr;
            job.output.resize(18 * 4);
            job.input = image;

            if (tensor == nullptr)
            {
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());

                if (use_multi_preprocess_stream_)
                {
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));

                    // owner = true, stream needs to be free during deconstruction
                    tensor->set_stream(preprocess_stream, true);
                }
                else
                {
                    preprocess_stream = stream_;

                    // owner = false, tensor ignored the stream
                    tensor->set_stream(preprocess_stream, false);
                }
            }
            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            auto workspace = tensor->get_workspace();
            uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_image);
            uint8_t *image_device = gpu_workspace;

            uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(size_image);
            uint8_t *image_host = cpu_workspace;

            // checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            //  speed up
            memcpy(image_host, image.data, size_image);
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::resize_bilinear_and_normalize(
                image_device, image.cols * 3, image.cols, image.rows,
                tensor->gpu<float>(), input_width_, input_height_,
                normalize_, preprocess_stream);
            return true;
        }

        virtual vector<shared_future<points>> commits(const vector<Mat> &images) override
        {
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<points> commit(const Mat &image) override
        {
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
        TRT::CUStream stream_ = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    };

    shared_ptr<Infer> create_infer(
        const string &engine_file, int gpuid)
    {
        shared_ptr<InferImpl> instance(new InferImpl());
        if (!instance->startup(engine_file, gpuid))
        {
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat &image, shared_ptr<TRT::Tensor> &tensor, int ibatch)
    {

        CUDAKernel::Norm normalize;
        float mean[] = {0.0, 0.0, 0.0};
        float std[] = {1.0, 1.0, 1.0};
        normalize = CUDAKernel::Norm::mean_std(
            mean, std, 1 / 255.0f, CUDAKernel::ChannelType::Invert);

        Size input_size(tensor->size(3), tensor->size(2));

        size_t size_image = image.cols * image.rows * 3;
        auto workspace = tensor->get_workspace();
        uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_image);
        uint8_t *image_device = gpu_workspace;

        uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(size_image);
        uint8_t *image_host = cpu_workspace;
        auto stream = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));

        CUDAKernel::resize_bilinear_and_normalize(
            image_device, image.cols * 3, image.cols, image.rows,
            tensor->gpu<float>(ibatch), input_size.width, input_size.height,
            normalize, stream);
        tensor->synchronize();
    }
};