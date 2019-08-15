#include <iostream>
#include <vector>

#include <boost/filesystem.hpp>

#include <librealsense2/rs.hpp>
#include <memory>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[]) {
    std::unique_ptr<rs2::context>  ctx  = std::unique_ptr<rs2::context>(new rs2::context());
    std::unique_ptr<rs2::pipeline> pipe = std::unique_ptr<rs2::pipeline>(new rs2::pipeline(*ctx));
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    rs2::device_list availableDevices = ctx->query_devices();

    if (availableDevices.size() == 0) {
        throw std::runtime_error("");
    }
    std::unique_ptr<rs2::device> device = std::unique_ptr<rs2::device>(new rs2::device(availableDevices.front()));

    int         width  = 640;
    int         height = 480;
    rs2::config config;
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, 30);
    config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGBA8, 30);

    rs2::pipeline_profile pipeline_profile = pipe->start(config);

    rs2::video_stream_profile depth_stream_profile = pipeline_profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2::video_stream_profile color_stream_profile = pipeline_profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    bool recording = false;

    int         current_idx = 0;
    std::string base_dir    = "/home/sliu/tmp/fusion/";

    while (true) {
        rs2::frameset frames = pipe->wait_for_frames();
        // frames = align_to_depth.process(frames);
        frames = align_to_color.process(frames);

        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();

        cv::Mat rgba_mat(color.get_height(), color.get_width(), CV_8UC4, (void *)color.get_data());
        cv::Mat depth_map(depth.get_height(), depth.get_width(), CV_16U, (void *)depth.get_data());

        cv::Mat rgb;
        cv::cvtColor(rgba_mat, rgb, cv::COLOR_RGBA2RGB);

        cv::imshow("rgb", rgba_mat);
        cv::imshow("depth", depth_map);

        if (recording) {
            cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
            cv::imwrite(base_dir + "c" + std::to_string(current_idx) + ".png", rgb);
            cv::imwrite(base_dir + "d" + std::to_string(current_idx) + ".png", depth_map);
            current_idx++;
        }

        int key = cv::waitKey(10);
        if (key == 's') {
            recording = true;
        }
        if (key == 'q') {
            break;
        }
    }

    rs2_intrinsics intrinsics_depth = depth_stream_profile.get_intrinsics();
    std::ofstream  out(base_dir + "info.txt");
    out << current_idx << std::endl;
    out << width << " " << height << std::endl;
    out << intrinsics_depth.fx << " " << intrinsics_depth.fy << " " << intrinsics_depth.ppx << " " << intrinsics_depth.ppy << std::endl;

    return 0;
}