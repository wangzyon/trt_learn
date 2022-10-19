#ifndef ROAD_HPP
#define ROAD_HPP

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace Road
{

    class Infer
    {
    public:
        virtual std::shared_future<cv::Mat> commit(const cv::Mat &image) = 0;
        virtual std::vector<std::shared_future<cv::Mat>> commits(const std::vector<cv::Mat> &images) = 0;
    };

    std::shared_ptr<Infer> create_infer(const std::string &engine_file, int gpuid);

    void image_to_tensor(const cv::Mat &image, std::shared_ptr<TRT::Tensor> &tensor, int ibatch);

}

#endif