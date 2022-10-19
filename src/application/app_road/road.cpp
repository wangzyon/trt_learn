#include "road.hpp"
#include "tensorRT/builder/trt_builder.hpp"
#include "tensorRT/common/ilogger.hpp"
#include "tensorRT/common/cuda_tools.hpp"
#include "tensorRT/common/preprocess_kernel.cuh"
#include "tensorRT/common/infer_controller.hpp"

#include <tuple>

namespace Road
{
    void decode_to_mask_invoker(const float *input, unsigned char *output, int edge, cudaStream_t stream);

    using InferControllerImpl = InferController<
        cv::Mat,                     // input
        cv::Mat,                     // output
        std::tuple<std::string, int> // start param
        >;

    class InferImpl : public Infer, InferControllerImpl
    {
    public:
        virtual bool startup(const std::string &file, int gpuid)
        {
            normalize_ = CUDAKernel::Norm::None();
            return InferController::startup(std::make_tuple(file, gpuid));
        }

        virtual void worker(std::promise<bool> &result)
        {
            std::string file;
            std::tie(file, gpu_) = start_param_;

            TRT::set_device(gpu_);
            auto engine = TRT::load_infer(file);
            if (engine == nullptr)
            {
                INFOE("Engine %s load failed.", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();
            int max_batch_size = engine->get_max_batch_size();
            auto input = engine->tensor("data");
            auto output = engine->tensor("tf.identity");

            input_width_ = input->size(3);
            input_height_ = input->size(2);
            tensor_allocator_ = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_ = engine->get_stream();
            result.set_value(true);

            TRT::Tensor mask({max_batch_size, input_height_, input_width_, 3}, TRT::DataType::UInt8);
            mask.to_gpu();

            std::vector<Job> fetch_jobs;
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
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {

                    auto &job = fetch_jobs[ibatch];
                    unsigned char *image_based_mask = mask.gpu<unsigned char>(ibatch);
                    float *image_based_output = output->gpu<float>(ibatch);
                    decode_to_mask_invoker(image_based_output, image_based_mask, input_width_ * input_height_, stream_);
                }

                mask.to_cpu();
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {

                    auto &job = fetch_jobs[ibatch];
                    unsigned char *image_based_output = mask.cpu<unsigned char>(ibatch);
                    job.pro->set_value(cv::Mat(input_height_, input_width_, CV_8UC3, image_based_output).clone());
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        };

        virtual bool preprocess(Job &job, const cv::Mat &image)
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

            if (tensor == nullptr)
            {
                tensor = std::make_shared<TRT::Tensor>();
                tensor->set_workspace(std::make_shared<TRT::MixMemory>());

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

            size_t image_size = image.cols * image.rows * 3;
            auto workspace = tensor->get_workspace();

            uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(image_size);
            uint8_t *image_device = gpu_workspace;

            uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(image_size);
            uint8_t *image_host = cpu_workspace;

            memcpy(image_host, image.data, image_size);
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, image_size, cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::resize_bilinear_and_normalize(
                image_device, image.cols * 3, image.cols, image.rows,
                tensor->gpu<float>(), input_width_, input_height_,
                normalize_, preprocess_stream);
            return true;
        };

        virtual std::vector<std::shared_future<cv::Mat>> commits(const std::vector<cv::Mat> &images) override
        {
            return InferControllerImpl::commits(images);
        }

        virtual std::shared_future<cv::Mat> commit(const cv::Mat &image) override
        {
            return InferControllerImpl::commit(image);
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
        TRT::CUStream stream_ = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    };

    std::shared_ptr<Infer> create_infer(const std::string &engine_file, int gpuid)
    {
        std::shared_ptr<InferImpl> instance(new InferImpl());
        if (!instance->startup(engine_file, gpuid))
        {
            instance.reset();
        }
        return instance;
    };

    void image_to_tensor(const cv::Mat &image, std::shared_ptr<TRT::Tensor> &tensor, int ibatch)
    {

        CUDAKernel::Norm normalize = CUDAKernel::Norm::None();
        cv::Size input_size(tensor->size(3), tensor->size(2));

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

}