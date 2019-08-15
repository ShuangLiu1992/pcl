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
void paint3DView(const KinfuTracker::View &rgb24, KinfuTracker::View &view, float colors_weight = 0.5f);
void mergePointNormal(const DeviceArray<PointXYZ> &cloud, const DeviceArray<Normal> &normals, DeviceArray<PointNormal> &output);
} // namespace gpu
} // namespace pcl

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MergedT, typename PointT>
typename pcl::PointCloud<MergedT>::Ptr merge(const pcl::PointCloud<PointT> &points, const pcl::PointCloud<pcl::RGB> &colors) {
    typename pcl::PointCloud<MergedT>::Ptr merged_ptr(new pcl::PointCloud<MergedT>());

    pcl::copyPointCloud(points, *merged_ptr);
    for (size_t i = 0; i < colors.size(); ++i)
        merged_ptr->points[i].rgba = colors.points[i].rgba;

    return merged_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const pcl::gpu::DeviceArray<pcl::PointXYZ> &triangles) {
    if (triangles.empty())
        return boost::shared_ptr<pcl::PolygonMesh>();

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width  = (int)triangles.size();
    cloud.height = 1;
    triangles.download(cloud.points);

    boost::shared_ptr<pcl::PolygonMesh> mesh_ptr(new pcl::PolygonMesh());
    pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);

    mesh_ptr->polygons.resize(triangles.size() / 3);
    for (size_t i = 0; i < mesh_ptr->polygons.size(); ++i) {
        pcl::Vertices v;
        v.vertices.push_back(i * 3 + 0);
        v.vertices.push_back(i * 3 + 2);
        v.vertices.push_back(i * 3 + 1);
        mesh_ptr->polygons[i] = v;
    }
    return mesh_ptr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView {
    pcl::gpu::KinfuTracker::View                  view_device_;
    pcl::gpu::KinfuTracker::View                  colors_device_;
    std::vector<pcl::gpu::KinfuTracker::PixelRGB> view_host_;
    pcl::gpu::RayCaster::Ptr                      raycaster_ptr_;
    pcl::gpu::KinfuTracker::DepthMap              generated_depth_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView {
    enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

    SceneCloudView() : extraction_mode_(GPU_Connected6), compute_normals_(false), valid_combined_(false) {
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

    void showMesh(pcl::gpu::KinfuTracker &kinfu, bool /*integrate_colors*/) {
        if (!marching_cubes_)
            marching_cubes_ = pcl::gpu::MarchingCubes::Ptr(new pcl::gpu::MarchingCubes());

        pcl::gpu::DeviceArray<pcl::PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);
        mesh_ptr_                                             = convertToMesh(triangles_device);
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

    boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};

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

    ImageView image_view_;

    pcl::gpu::KinfuTracker kinfu_(height, width);
    kinfu_.initColorIntegration(2);
    kinfu_.setDepthIntrinsics(fx, fy, cx, cy);
    image_view_.raycaster_ptr_ = pcl::gpu::RayCaster::Ptr(new pcl::gpu::RayCaster(kinfu_.rows(), kinfu_.cols(), fx, fy, cx, cy));

    pcl::gpu::KinfuTracker::DepthMap                            depth_device_;
    pcl::gpu::PtrStepSz<const unsigned short>                   depth_;
    pcl::gpu::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> rgb24_;

    SceneCloudView                                scene_cloud_view_;
    pcl::gpu::KinfuTracker::View                  view_device_;
    pcl::gpu::KinfuTracker::View                  colors_device_;
    std::vector<pcl::gpu::KinfuTracker::PixelRGB> view_host_;

    pcl::gpu::RayCaster::Ptr raycaster_ptr_;

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
            image_view_.colors_device_.upload(rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
            kinfu_(depth_device_, image_view_.colors_device_);
            kinfu_.getImage(view_device_);

            colors_device_.upload(rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
            paint3DView(colors_device_, view_device_, 0.5);

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
    const SceneCloudView &view = scene_cloud_view_;
    pcl::io::savePLYFileASCII("/home/sliu/Dropbox/sync/mesh/cloud.ply",
                              *merge<pcl::PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
    // pcl::io::savePLYFile("mesh.ply", mesh);

    return 0;
}