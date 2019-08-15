#include <iostream>
#include <vector>

#include <boost/filesystem.hpp>

#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/kinfu/raycaster.h>

#include <librealsense2/rs.hpp>
#include <memory>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "evaluation.h"

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename CloudPtr> void writeCloudFile(const CloudPtr &cloud_prt) { pcl::io::savePLYFileASCII("cloud.ply", *cloud_prt); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void writePolygonMeshFile(const pcl::PolygonMesh &mesh) { pcl::io::savePLYFile("mesh.ply", mesh); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    int device_id = 0;
    pcl::gpu::setDevice(device_id);
    pcl::gpu::printShortCudaDeviceInfo(device_id);
    std::vector<float> depth_intrinsics;

    std::unique_ptr<rs2::context>  ctx;
    std::unique_ptr<rs2::device>   device;
    std::unique_ptr<rs2::pipeline> pipe;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    ctx = std::unique_ptr<rs2::context>(new rs2::context());

    rs2::device_list availableDevices = ctx->query_devices();

    printf("There are %d connected RealSense devices.\n", availableDevices.size());
    if (availableDevices.size() == 0) {
        ctx.reset();
    }

    device = std::unique_ptr<rs2::device>(new rs2::device(availableDevices.front()));

    pipe = std::unique_ptr<rs2::pipeline>(new rs2::pipeline(*ctx));

    rs2::config config;
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGBA8, 30);

    auto availableSensors = device->query_sensors();
    std::cout << "Device consists of " << availableSensors.size() << " sensors:" << std::endl;
    for (rs2::sensor sensor : availableSensors) {
        // print_sensor_information(sensor);

        if (rs2::depth_sensor dpt_sensor = sensor.as<rs2::depth_sensor>()) {
            float scale = dpt_sensor.get_depth_scale();
            std::cout << "Scale factor for depth sensor is: " << scale << std::endl;
        }
    }

    rs2::pipeline_profile pipeline_profile = pipe->start(config);

    rs2::video_stream_profile depth_stream_profile = pipeline_profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2::video_stream_profile color_stream_profile = pipeline_profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    pcl::gpu::KinfuTracker kinfu_;

    // - intrinsics
    rs2_intrinsics intrinsics_depth = depth_stream_profile.get_intrinsics();
    rs2_intrinsics intrinsics_rgb   = color_stream_profile.get_intrinsics();
    rs2_extrinsics rs_extrinsics    = color_stream_profile.get_extrinsics_to(depth_stream_profile);
    kinfu_.setDepthIntrinsics(intrinsics_depth.fx, intrinsics_depth.fy, intrinsics_depth.ppx, intrinsics_depth.ppy);

    int currentIndex = 0;

    ImageView       image_view_;
    Evaluation::Ptr evaluation_ptr_;
    evaluation_ptr_ = Evaluation::Ptr(new Evaluation("/home/sliu/data/fusion/rgbd_dataset_freiburg1_xyz/"));
    evaluation_ptr_->setMatchFile("matches.txt");

    kinfu_.initColorIntegration(2);
    kinfu_.setDepthIntrinsics(evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
    image_view_.raycaster_ptr_ = pcl::gpu::RayCaster::Ptr(new pcl::gpu::RayCaster(
        kinfu_.rows(), kinfu_.cols(), evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy));

    pcl::gpu::KinfuTracker::DepthMap                            depth_device_;
    pcl::gpu::PtrStepSz<const unsigned short>                   depth_;
    pcl::gpu::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> rgb24_;

    SceneCloudView                                scene_cloud_view_;
    pcl::gpu::KinfuTracker::View                  view_device_;
    pcl::gpu::KinfuTracker::View                  colors_device_;
    std::vector<pcl::gpu::KinfuTracker::PixelRGB> view_host_;

    pcl::gpu::RayCaster::Ptr raycaster_ptr_;

    pcl::gpu::KinfuTracker::DepthMap generated_depth_;
    while (evaluation_ptr_->grab(currentIndex, depth_, rgb24_)) {
        rs2::frameset frames = pipe->wait_for_frames();
        // frames = align_to_depth.process(frames);
        frames = align_to_color.process(frames);

        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();

        cv::Mat rgba(color.get_height(), color.get_width(), CV_8UC4, (void *)color.get_data());
        cv::Mat rgb;
        cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);

        memcpy(const_cast<unsigned short *>(depth_.data), depth.get_data(),
               depth.get_width() * depth.get_height() * sizeof(unsigned short));
        memcpy(const_cast<pcl::gpu::KinfuTracker::PixelRGB *>(rgb24_.data), rgb.data,
               color.get_width() * color.get_height() * sizeof(pcl::gpu::KinfuTracker::PixelRGB));

        {
            const pcl::gpu::PtrStepSz<const unsigned short> &                  depth = depth_;
            const pcl::gpu::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> &rgb24 = rgb24_;
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            image_view_.colors_device_.upload(rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
            kinfu_(depth_device_, image_view_.colors_device_);
            // image_view_.showDepth(depth);
            // image_view_.showGeneratedDepth(kinfu_, kinfu_.getCameraPose());

            // scene_cloud_view_.showMesh(kinfu_, true);
            kinfu_.getImage(view_device_);

            colors_device_.upload(rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
            paint3DView(colors_device_, view_device_, 1);

            int cols;
            view_device_.download(view_host_, cols);
            cv::Mat rgbb(view_device_.rows(), view_device_.cols(), CV_8UC3, &view_host_[0]);
            cv::cvtColor(rgbb, rgbb, cv::COLOR_RGB2BGR);
            cv::imshow("rgb_image", rgbb);
            char key = cv::waitKey(10);
            if (key == 'q') {
                break;
            }
            if (key == 'r') {
                kinfu_.reset();
            }
        }

        currentIndex += 1;
    }
    scene_cloud_view_.show(kinfu_, true);
    const SceneCloudView &view = scene_cloud_view_;
    if (view.point_colors_ptr_->points.empty()) // no colors
    {
        if (view.valid_combined_)
            writeCloudFile(view.combined_ptr_);
        else
            writeCloudFile(view.cloud_ptr_);
    } else {
        if (view.valid_combined_)
            writeCloudFile(merge<pcl::PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
        else
            writeCloudFile(merge<pcl::PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
    }

    return 0;
}
