#include <iostream>
#include <vector>

#include <boost/filesystem.hpp>

#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/kinfu/raycaster.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../src/internal.h"

namespace pcl {
namespace gpu {
void mergePointNormal(const DeviceArray<PointXYZ> &cloud, const DeviceArray<Normal> &normals, DeviceArray<PointNormal> &output);
} // namespace gpu
} // namespace pcl

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView {
    enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

    SceneCloudView() : extraction_mode_(GPU_Connected6), compute_normals_(true), valid_combined_(false) {
        cloud_ptr_        = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        normals_ptr_      = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
        combined_ptr_     = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>);
        point_colors_ptr_ = pcl::PointCloud<pcl::RGB>::Ptr(new pcl::PointCloud<pcl::RGB>);
    }

    void show(pcl::gpu::KinfuTracker &kinfu, bool integrate_colors) {
        valid_combined_ = false;

        if (extraction_mode_ != GPU_Connected6) // So use CPU
        {
            kinfu.volume().fetchCloudHost(*cloud_ptr_, extraction_mode_ == CPU_Connected26);
        } else {
            pcl::gpu::DeviceArray<pcl::PointXYZ> extracted = kinfu.volume().fetchCloud(cloud_buffer_device_);

            if (compute_normals_) {
                kinfu.volume().fetchNormals(extracted, normals_device_);
                pcl::gpu::mergePointNormal(extracted, normals_device_, combined_device_);
                combined_device_.download(combined_ptr_->points);
                combined_ptr_->width  = (int)combined_ptr_->points.size();
                combined_ptr_->height = 1;

                valid_combined_ = true;
            } else {
                extracted.download(cloud_ptr_->points);
                cloud_ptr_->width  = (int)cloud_ptr_->points.size();
                cloud_ptr_->height = 1;
            }

            if (integrate_colors) {
                kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
                point_colors_device_.download(point_colors_ptr_->points);
                point_colors_ptr_->width  = (int)point_colors_ptr_->points.size();
                point_colors_ptr_->height = 1;
            } else
                point_colors_ptr_->points.clear();
        }
        size_t points_size = valid_combined_ ? combined_ptr_->points.size() : cloud_ptr_->points.size();
    }

    int  extraction_mode_;
    bool compute_normals_;
    bool valid_combined_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_;
    pcl::PointCloud<pcl::Normal>::Ptr   normals_ptr_;

    pcl::gpu::DeviceArray<pcl::PointXYZ> cloud_buffer_device_;
    pcl::gpu::DeviceArray<pcl::Normal>   normals_device_;

    pcl::PointCloud<pcl::PointNormal>::Ptr  combined_ptr_;
    pcl::gpu::DeviceArray<pcl::PointNormal> combined_device_;

    pcl::gpu::DeviceArray<pcl::RGB> point_colors_device_;
    pcl::PointCloud<pcl::RGB>::Ptr  point_colors_ptr_;

    pcl::gpu::MarchingCubes::Ptr         marching_cubes_;
    pcl::gpu::DeviceArray<pcl::PointXYZ> triangles_buffer_device_;
    pcl::gpu::DeviceArray<pcl::PointXYZ> colors_buffer_device_;

    boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};

void save_dump(std::string output_path, unsigned int n_points, const pcl::PointCloud<pcl::PointNormal> &cloud,
               const pcl::PointCloud<pcl::RGB> &colors) {
    std::ofstream stream(output_path, std::ios::binary);

    int                       threshold       = 120;
    int                       color_threshold = 1;
    std::vector<unsigned int> valid_list;
    for (uint i = 0; i < n_points; i++) {
        if (cloud.points[i].data[3] > threshold && colors.points[i].a > color_threshold) {
            if (!std::isnan(cloud.points[i].normal_x) && !std::isnan(cloud.points[i].normal_y) && !std::isnan(cloud.points[i].normal_z)) {
                valid_list.push_back(i);
            }
        }
    }

    unsigned int valid_size = valid_list.size();
    stream.write(reinterpret_cast<const char *>(&valid_size), sizeof(valid_size));

    for (size_t idx = 0; idx < valid_list.size(); idx++) {
        size_t i = valid_list[idx];
        uchar  co[3];
        co[0] = colors.points[i].r;
        co[1] = colors.points[i].g;
        co[2] = colors.points[i].b;
        float po[3];
        po[0] = cloud.points[i].x;
        po[1] = cloud.points[i].y;
        po[2] = cloud.points[i].z;
        stream.write(reinterpret_cast<const char *>(po), sizeof(float) * 3);
        po[0] = cloud.points[i].normal_x;
        po[1] = cloud.points[i].normal_y;
        po[2] = cloud.points[i].normal_z;
        stream.write(reinterpret_cast<const char *>(po), sizeof(float) * 3);
        stream.write(reinterpret_cast<const char *>(co), sizeof(uchar) * 3);
    }
}

int main(int argc, char *argv[]) {
    std::string   base_dir = "/home/sliu/tmp/fusion/";
    std::ifstream in(base_dir + "info.txt");
    int           total_frames;
    int           width;
    int           height;
    in >> total_frames;
    in >> width;
    in >> height;
    float fx;
    float fy;
    float cx;
    float cy;
    in >> fx;
    in >> fy;
    in >> cx;
    in >> cy;

    std::cout << total_frames << std::endl;
    std::cout << width << " " << height << std::endl;
    std::cout << fx << " " << fy << " " << cx << " " << cy << std::endl;

    pcl::gpu::KinfuTracker kinfu_(height, width);
    kinfu_.initColorIntegration(2);
    kinfu_.setDepthIntrinsics(fx, fy, cx, cy);

    pcl::gpu::KinfuTracker::DepthMap                            depth_device_;
    pcl::gpu::PtrStepSz<const unsigned short>                   depth_;
    pcl::gpu::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> rgb24_;

    SceneCloudView                                scene_cloud_view_;
    pcl::gpu::KinfuTracker::View                  view_device_;
    pcl::gpu::KinfuTracker::View                  colors_device_;
    std::vector<pcl::gpu::KinfuTracker::PixelRGB> view_host_;

    pcl::gpu::KinfuTracker::DepthMap generated_depth_;
    for (int i = 0; i < total_frames; i++) {
        cv::Mat rgb   = cv::imread(base_dir + "c" + std::to_string(i) + ".png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        cv::Mat depth = cv::imread(base_dir + "d" + std::to_string(i) + ".png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

        depth_.data = reinterpret_cast<unsigned short *>(depth.data);
        depth_.cols = depth.cols;
        depth_.rows = depth.rows;
        depth_.step = depth.cols * sizeof(unsigned short);

        rgb24_.data = reinterpret_cast<pcl::gpu::KinfuTracker::PixelRGB *>(rgb.data);
        rgb24_.cols = rgb.cols;
        rgb24_.rows = rgb.rows;
        rgb24_.step = rgb.cols * sizeof(pcl::gpu::KinfuTracker::PixelRGB);

        {
            const pcl::gpu::PtrStepSz<const unsigned short> &                  depth = depth_;
            const pcl::gpu::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> &rgb24 = rgb24_;
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            colors_device_.upload(rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
            kinfu_(depth_device_, colors_device_);
            kinfu_.getImage(view_device_);

            colors_device_.upload(rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
            pcl::device::paint3DView(colors_device_, view_device_, 0.5);

            int cols;
            view_device_.download(view_host_, cols);
            cv::Mat rgbb(view_device_.rows(), view_device_.cols(), CV_8UC3, &view_host_[0]);
            cv::cvtColor(rgbb, rgbb, cv::COLOR_BGR2RGB);
            cv::imshow("rgb_image", rgbb);
            char key = cv::waitKey(10);
            if (key == 'q') {
                break;
            }
            if (key == 'r') {
                kinfu_.reset();
            }
        }
    }

    scene_cloud_view_.show(kinfu_, true);
    save_dump("/home/sliu/Dropbox/sync/mesh/test.dump", scene_cloud_view_.combined_ptr_->size(), *scene_cloud_view_.combined_ptr_,
              *scene_cloud_view_.point_colors_ptr_);

    return 0;
}