#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include "tensorRT/common/ilogger.hpp"
#include "tensorRT/builder/trt_builder.hpp"
#include "app_road/road.hpp"
#include "app_ldrn/ldrn.hpp"
#include "app_lane/lane.hpp"
#include "app_yolo/yolo.hpp"
#include <functional>

#define SELF_DRIVING_WORKSPACE "self_driving"
#define SELF_DRIVING_DEVICE 0

static bool exists(const std::string &path)
{

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

static const char *cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i)
    {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
        r = v;
        g = p;
        b = q;
        break;
    default:
        r = 1;
        g = 1;
        b = 1;
        break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    ;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

#define __create_infer(INFER_NAMESPACE, model_name, mode) create_infer_invoker<INFER_NAMESPACE::Infer>(INFER_NAMESPACE::create_infer, model_name, mode)

// ##INFER_NAMESPACE之前有逗号、分号、空格时，##应省略
#define int8process_func(INFER_NAMESPACE)                                                                    \
    [=](int current, int count, const std::vector<std::string> &files, std::shared_ptr<TRT::Tensor> &tensor) \
    {                                                                                                        \
        INFO("%s Int8 %d / %d", current, count);                                                             \
        for (int i = 0; i < files.size(); ++i)                                                               \
        {                                                                                                    \
            auto image = cv::imread(files[i]);                                                               \
            INFER_NAMESPACE::image_to_tensor(image, tensor, i);                                              \
        }                                                                                                    \
    }

static bool build_model_invoker(TRT::Int8Process int8process, const std::string &model_name, TRT::Mode mode, int max_batch_size)
{
    std::string onnx_model_file = iLogger::format("%s/model/%s.onnx", SELF_DRIVING_WORKSPACE, model_name.c_str());
    std::string trt_model_file = iLogger::format("%s/model/%s.%s.trtmodel", SELF_DRIVING_WORKSPACE, model_name.c_str(), TRT::mode_string(mode));
    std::string int8ImageDirectory = iLogger::format("%s/media", SELF_DRIVING_WORKSPACE);

    if (!exists(trt_model_file))
    {
        INFO("Compile %s...", trt_model_file.c_str());
        TRT::compile(mode,
                     max_batch_size,
                     onnx_model_file,
                     trt_model_file,
                     {},
                     int8process,
                     int8ImageDirectory,
                     "",
                     1ul << 30);
    }
    return true;
}

static bool build_model(TRT::Mode mode, int max_batch_size)
{
    bool success = true;
    success = success && build_model_invoker(int8process_func(Yolo), "yolox_tiny_416x416", mode, max_batch_size);
    success = success && build_model_invoker(int8process_func(Lane), "lane_detection_288_800", mode, max_batch_size);
    success = success && build_model_invoker(int8process_func(Road), "road_segmentation_512x896", mode, max_batch_size);
    success = success && build_model_invoker(int8process_func(Ldrn), "ldrn_256x512", mode, max_batch_size);
    return success;
}

template <typename R>
static std::shared_ptr<R> create_infer_invoker(std::function<std::shared_ptr<R>(const std::string &, int device_id)> create_infer_fptr, const std::string &model_name, TRT::Mode mode)
{
    std::string trt_model_file = iLogger::format("%s/model/%s.%s.trtmodel", SELF_DRIVING_WORKSPACE, model_name.c_str(), TRT::mode_string(mode));
    std::shared_ptr<R> infer = create_infer_fptr(trt_model_file, SELF_DRIVING_DEVICE);
    if (infer == nullptr)
    {
        INFOF("create %s infer failed.", model_name.c_str());
    }
    INFO("create %s infer success.", model_name.c_str());
    return infer;
}

static void inference(TRT::Mode mode)
{
    auto yolo_infer = __create_infer(Yolo, "yolox_tiny_416x416", mode);
    auto lane_infer = __create_infer(Lane, "lane_detection_288_800", mode);
    auto road_infer = __create_infer(Road, "road_segmentation_512x896", mode);
    auto ldrn_infer = __create_infer(Ldrn, "ldrn_256x512", mode);

    std::string imageDirectory = iLogger::format("%s/media", SELF_DRIVING_WORKSPACE);
    auto files = iLogger::find_files(imageDirectory, "*.jpg");
    std::vector<cv::Mat> images;
    for (int i = 0; i < files.size(); ++i)
    {
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    auto yolo_futures = yolo_infer->commits(images);
    auto lane_futures = lane_infer->commits(images);
    auto road_futures = road_infer->commits(images);

    auto begin_timer = iLogger::timestamp_now_float();
    for (int i = 0; i < 100; i++)
    {
        auto yolo_futures = yolo_infer->commits(images);
        auto lane_futures = lane_infer->commits(images);
        auto road_futures = road_infer->commits(images);
    }
    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / 100 / 16;
    INFO("%s average: %.2f ms / image, FPS: %.2f", "", inference_average_time, 1000 / inference_average_time);
}

static void merge_images(
    const cv::Mat &image, const cv::Mat &road,
    const cv::Mat &depth, cv::Mat &scence)
{
    image.copyTo(scence(cv::Rect(0, 0, image.cols, image.rows)));

    auto road_crop = road(cv::Rect(0, road.rows * 0.5, road.cols, road.rows * 0.5));
    std::cout << image.cols << "  " << image.rows << std::endl;
    std::cout << road_crop.cols << "  " << road_crop.rows << std::endl;
    std::cout << scence.cols << "  " << scence.rows << std::endl;
    road_crop.copyTo(scence(cv::Rect(0, image.rows, road_crop.cols, road_crop.rows)));

    // auto depth_crop = depth(cv::Rect(0, depth.rows * 0.18, depth.cols, depth.rows * (1 - 0.18)));
    // depth_crop.copyTo(scence(cv::Rect(image.cols, image.rows * 0.25, depth_crop.cols, depth_crop.rows)));
}
static void inferenceV2(TRT::Mode mode)
{
    auto yolo_infer = __create_infer(Yolo, "yolox_tiny_416x416", mode);
    auto lane_infer = __create_infer(Lane, "lane_detection_288_800", mode);
    auto road_infer = __create_infer(Road, "road_segmentation_512x896", mode);
    auto ldrn_infer = __create_infer(Ldrn, "ldrn_256x512", mode);

    cv::Mat image = cv::imread("/volume/wzy/project/tensorRT_Zoo/workspace/self_driving/media/test_image_04.jpg");

    auto yolo_future = yolo_infer->commit(image);
    auto lane_future = lane_infer->commit(image);
    auto road_future = road_infer->commit(image);
    auto ldrn_future = ldrn_infer->commit(image);

    // plot yolo
    auto &boxes = yolo_future.get();
    for (auto &obj : boxes)
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 2);

        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width;
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 23), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 0.6, cv::Scalar::all(0), 2, 8);
    }

    // plot lane
    auto &points = lane_future.get();
    for (int j = 0; j < points.size(); j++)
    {
        cv::circle(image, points[j], 5, cv::Scalar(0, 255, 0), -1);
    }

    cv::Mat road_image = road_future.get();
    cv::Mat ldrn_image = ldrn_future.get();
    cv::Mat scence = cv::Mat(image.rows * 1.5, image.cols * 2, CV_8UC3, cv::Scalar::all(0));
    merge_images(image, road_image, ldrn_image, scence);
    cv::imwrite("/volume/wzy/project/tensorRT_Zoo/output.jpg", scence);
}

template <typename T>
static void inference_and_performance_invoker(std::shared_ptr<T> infer, const std::string &model_name, TRT::Mode mode, std::vector<cv::Mat> &images, int ntest = 100)
{
    auto begin_timer = iLogger::timestamp_now_float();
    for (int i = 0; i < ntest; ++i)
    {
        auto futures = infer->commits(images);
    }

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto mode_name = TRT::mode_string(mode);
    INFO("%s average: %.2f ms / image, FPS: %.2f", model_name.c_str(), inference_average_time, 1000 / inference_average_time);
}

static void inference_and_performance(TRT::Mode mode)
{
    auto yolo_infer = __create_infer(Yolo, "yolox_tiny_416x416", mode);
    auto lane_infer = __create_infer(Lane, "lane_detection_288_800", mode);
    auto road_infer = __create_infer(Road, "road_segmentation_512x896", mode);
    auto ldrn_infer = __create_infer(Ldrn, "ldrn_256x512", mode);

    std::string imageDirectory = iLogger::format("%s/media", SELF_DRIVING_WORKSPACE);
    auto files = iLogger::find_files(imageDirectory, "*.jpg");
    std::vector<cv::Mat> images;
    for (int i = 0; i < files.size(); ++i)
    {
        auto image = cv::imread(files[i]);
        images.push_back(image);
    }

    // warmup
    inference_and_performance_invoker<Yolo::Infer>(yolo_infer, "yolox_tiny_416x416", mode, images, 100);

    // performance test
    inference_and_performance_invoker<Lane::Infer>(lane_infer, "lane_detection_288_800", mode, images, 100);
    // performance test
    inference_and_performance_invoker<Road::Infer>(road_infer, "road_segmentation_512x896", mode, images, 100);
    // performance test
    inference_and_performance_invoker<Yolo::Infer>(yolo_infer, "yolox_tiny_416x416", mode, images, 100);
    // performance test
    // inference_and_performance_invoker<Ldrn::Infer>(ldrn_infer, "ldrn_256x512", mode, images, 100);
}

int main(int argc, char **argv)
{
    TRT::Mode mode = TRT::Mode::FP16;
    int max_batch_size = 8;

    build_model(mode, max_batch_size);
    // inference(mode);
    inferenceV2(mode);
    // inference_and_performance(mode);
    return 0;
}

// static void inference()
// {

//     // auto image = cv::imread("imgs/dashcam_00.jpg");
//     auto yolov5 = Yolo::create_infer("yolov5s.trtmodel", Yolo::Type::V5, 0, 0.25, 0.45);
//     auto road = Road::create_infer("road-segmentation-adas.trtmodel", 0);
//     auto ldrn = Ldrn::create_infer("ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel", 0);
//     auto lane = Lane::create_infer("new-lane.trtmodel", 0);

//     cv::Mat image, scence;
//     cv::VideoCapture cap("4k-tokyo-drive-thru-ikebukuro.mp4");
//     float fps = cap.get(cv::CAP_PROP_FPS);
//     int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
//     int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//     scence = cv::Mat(height * 1.5, width * 2, CV_8UC3, cv::Scalar::all(0));
//     cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('M', 'P', 'G', '2'), fps, scence.size());
//     // auto scence = cv::Mat(image.rows * 1.5, image.cols * 2, CV_8UC3, cv::Scalar::all(0));

//     while (cap.read(image))
//     {
//         auto roadmask_fut = road->commit(image);
//         auto boxes_fut = yolov5->commit(image);
//         auto depth_fut = ldrn->commit(image);
//         auto points_fut = lane->commit(image);
//         auto roadmask = roadmask_fut.get();
//         auto boxes = boxes_fut.get();
//         auto depth = depth_fut.get();
//         auto points = points_fut.get();
//         cv::resize(depth, depth, image.size());
//         cv::resize(roadmask, roadmask, image.size());

//         for (auto &box : boxes)
//         {
//             int cx = (box.left + box.right) * 0.5 + 0.5;
//             int cy = (box.top + box.bottom) * 0.5 + 0.5;
//             float distance = depth.at<float>(cy, cx) / 5;
//             if (fabs(cx - (image.cols * 0.5)) <= 200 && cy >= image.rows * 0.85)
//                 continue;

//             cv::Scalar color(0, 255, 0);
//             cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

//             auto name = cocolabels[box.class_label];
//             auto caption = cv::format("%s %.2f", name, distance);
//             int text_width = cv::getTextSize(caption, 0, 0.5, 1, nullptr).width + 10;
//             cv::rectangle(image, cv::Point(box.left - 3, box.top - 20), cv::Point(box.left + text_width, box.top), color, -1);
//             cv::putText(image, caption, cv::Point(box.left, box.top - 5), 0, 0.5, cv::Scalar::all(0), 1, 16);
//         }

//         cv::Scalar colors[] = {
//             cv::Scalar(255, 0, 0),
//             cv::Scalar(0, 0, 255),
//             cv::Scalar(0, 0, 255),
//             cv::Scalar(255, 0, 0)};
//         for (int i = 0; i < 18; ++i)
//         {
//             for (int j = 0; j < 4; ++j)
//             {
//                 auto &p = points[i * 4 + j];
//                 if (p.x > 0)
//                 {
//                     auto color = colors[j];
//                     cv::circle(image, p, 5, color, -1, 16);
//                 }
//             }
//         }
//         merge_images(image, roadmask, to_render_depth(depth), scence);
//         // cv::imwrite("merge.jpg", scence);

//         writer.write(scence);
//         INFO("Process");
//     }
//     writer.release();
// }

// std::vector<std::shared_future<ObjectDetector::BoxArray>> yolo_futures = yolo_infer->commits(images);
//     for (int i = 0; i < yolo_futures.size(); i++)
//     {
//         auto boxes = yolo_futures[i].get();
//         for (auto &obj : boxes)
//         {
//             uint8_t b, g, r;
//             std::tie(b, g, r) = random_color(obj.class_label);
//             cv::rectangle(images[i], cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 2);

//             auto name = cocolabels[obj.class_label];
//             auto caption = cv::format("%s %.2f", name, obj.confidence);
//             int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width;
//             cv::rectangle(images[i], cv::Point(obj.left - 3, obj.top - 23), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
//             cv::putText(images[i], caption, cv::Point(obj.left, obj.top - 5), 0, 0.6, cv::Scalar::all(0), 2, 8);
//         }
//     }

//     // lane
//     std::string lane_trt_model_file = iLogger::format("%s/model/lane_detection_288_800.%s.trtmodel", SELF_DRIVING_WORKSPACE, TRT::mode_string(mode));
//     auto lane_infer = Lane::create_infer(lane_trt_model_file, device_id);
//     if (lane_infer == nullptr)
//     {
//         INFOE("create lane infer failed.");
//         return;
//     }

//     std::vector<std::shared_future<cv::Mat>> road_futures = road_infer->commits(images);
//     for (int i = 0; i < road_futures.size(); i++)
//     {
//         auto road_image = road_futures[i].get();
//         std::string out_file = iLogger::format("%s/output/test_image_%d_road_infer.jpg", SELF_DRIVING_WORKSPACE, i);
//         cv::imwrite(out_file, road_image);
//     }

//     // for (int i = 0; i < images.size(); i++)
//     // {
//     //     std::string out_file = iLogger::format("%s/output/test_image_%d_infer.jpg", SELF_DRIVING_WORKSPACE, i);
//     //     cv::imwrite(out_file, images[i]);
//     // }DEVICE

// static cv::Mat &yolo_inference(Yolo::Infer *infer, cv::Mat &image, bool draw = false)
// {
//     std::shared_future<ObjectDetector::BoxArray> future = infer->commit(image);
//     auto &boxes = future.get();
//     if (draw)
//     {
//         for (auto &obj : boxes)
//         {
//             uint8_t b, g, r;
//             std::tie(b, g, r) = random_color(obj.class_label);
//             cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 2);

//             auto name = cocolabels[obj.class_label];
//             auto caption = cv::format("%s %.2f", name, obj.confidence);
//             int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width;
//             cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 23), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
//             cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 0.6, cv::Scalar::all(0), 2, 8);
//         }
//     }
//     return image;
// }

// static cv::Mat &lane_inference(Lane::Infer *infer, cv::Mat &image, bool draw = false)
// {
//     std::shared_future<std::vector<cv::Point2f>> future = infer->commit(image);
//     auto &points = future.get();
//     if (draw)
//     {
//         for (int j = 0; j < points.size(); j++)
//         {
//             cv::circle(image, points[j], 5, cv::Scalar(0, 255, 0), -1);
//         }
//     }
//     return image;
// }

// static cv::Mat road_inference(Road::Infer *infer, cv::Mat &image)
// {
//     std::shared_future<cv::Mat> future = infer->commit(image);
//     cv::Mat out_image = future.get();
//     return out_image;
// }

// static cv::Mat ldrn_inference(Ldrn::Infer *infer, cv::Mat &image)
// {
//     std::shared_future<cv::Mat> future = infer->commit(image);
//     cv::Mat out_image = future.get();
//     return out_image;
// }