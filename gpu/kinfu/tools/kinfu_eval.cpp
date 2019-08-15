#include <iostream>
#include <vector>

#include <boost/filesystem.hpp>

#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/kinfu/raycaster.h>

#include <librealsense2/rs.hpp>
#include <memory>

#include <pcl/common/time.h>
#include <pcl/exceptions.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "evaluation.h"
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <pcl/common/angles.h>

#include "tsdf_volume.h"
#include "tsdf_volume.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using ScopeTimeT = pcl::ScopeTime;

#include "../src/internal.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace pcl {
namespace gpu {
void paint3DView(const KinfuTracker::View &rgb24, KinfuTracker::View &view, float colors_weight = 0.5f);
void mergePointNormal(const DeviceArray<PointXYZ> &cloud, const DeviceArray<Normal> &normals, DeviceArray<PointNormal> &output);
} // namespace gpu

namespace visualization {
//////////////////////////////////////////////////////////////////////////////////////
/** \brief RGB handler class for colors. Uses the data present in the "rgb" or "rgba"
 * fields from an additional cloud as the color at each point.
 * \author Anatoly Baksheev
 * \ingroup visualization
 */
template <typename PointT> class PointCloudColorHandlerRGBCloud : public PointCloudColorHandler<PointT> {
    using PointCloudColorHandler<PointT>::capable_;
    using PointCloudColorHandler<PointT>::cloud_;

    using PointCloudConstPtr = typename PointCloudColorHandler<PointT>::PointCloud::ConstPtr;
    using RgbCloudConstPtr   = pcl::PointCloud<RGB>::ConstPtr;

  public:
    using Ptr      = boost::shared_ptr<PointCloudColorHandlerRGBCloud<PointT>>;
    using ConstPtr = boost::shared_ptr<const PointCloudColorHandlerRGBCloud<PointT>>;

    /** \brief Constructor. */
    PointCloudColorHandlerRGBCloud(const PointCloudConstPtr &cloud, const RgbCloudConstPtr &colors) : rgb_(colors) {
        cloud_   = cloud;
        capable_ = true;
    }

    /** \brief Obtain the actual color for the input dataset as vtk scalars.
     * \param[out] scalars the output scalars containing the color for the dataset
     * \return true if the operation was successful (the handler is capable and
     * the input cloud was given as a valid pointer), false otherwise
     */
    bool getColor(vtkSmartPointer<vtkDataArray> &scalars) const override {
        if (!capable_ || !cloud_)
            return (false);

        if (!scalars)
            scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);

        vtkIdType nr_points = vtkIdType(cloud_->points.size());
        reinterpret_cast<vtkUnsignedCharArray *>(&(*scalars))->SetNumberOfTuples(nr_points);
        unsigned char *colors = reinterpret_cast<vtkUnsignedCharArray *>(&(*scalars))->GetPointer(0);

        // Color every point
        if (nr_points != int(rgb_->points.size()))
            std::fill(colors, colors + nr_points * 3, static_cast<unsigned char>(0xFF));
        else
            for (vtkIdType cp = 0; cp < nr_points; ++cp) {
                int idx         = cp * 3;
                colors[idx + 0] = rgb_->points[cp].r;
                colors[idx + 1] = rgb_->points[cp].g;
                colors[idx + 2] = rgb_->points[cp].b;
            }
        return (true);
    }

  private:
    std::string getFieldName() const override { return ("additional rgb"); }
    std::string getName() const override { return ("PointCloudColorHandlerRGBCloud"); }

    RgbCloudConstPtr rgb_;
};
} // namespace visualization
} // namespace pcl

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch {
    enum { EACH = 33 };
    SampledScopeTime(int &time_ms) : time_ms_(time_ms) {}
    ~SampledScopeTime() {
        static int                      i_         = 0;
        static boost::posix_time::ptime starttime_ = boost::posix_time::microsec_clock::local_time();
        time_ms_ += getTime();
        if (i_ % EACH == 0 && i_) {
            boost::posix_time::ptime endtime_ = boost::posix_time::microsec_clock::local_time();
            cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )"
                 << "( real: " << 1000.f * EACH / (endtime_ - starttime_).total_milliseconds() << "fps )" << endl;
            time_ms_   = 0;
            starttime_ = endtime_;
        }
        ++i_;
    }

  private:
    int &time_ms_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CloudT> void writeCloudFile(int format, const CloudT &cloud);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void writePolygonMeshFile(int format, const pcl::PolygonMesh &mesh);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT> &points, const PointCloud<RGB> &colors) {
    typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());

    pcl::copyPointCloud(points, *merged_ptr);
    for (size_t i = 0; i < colors.size(); ++i)
        merged_ptr->points[i].rgba = colors.points[i].rgba;

    return merged_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ> &triangles) {
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

struct CurrentFrameCloudView {
    CurrentFrameCloudView() : cloud_device_(480, 640), cloud_viewer_("Frame Cloud Viewer") {
        cloud_ptr_ = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);

        cloud_viewer_.setBackgroundColor(0, 0, 0.15);
        cloud_viewer_.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1);
        cloud_viewer_.addCoordinateSystem(1.0, "global");
        cloud_viewer_.initCameraParameters();
        cloud_viewer_.setPosition(0, 500);
        cloud_viewer_.setSize(640, 480);
        cloud_viewer_.setCameraClipDistances(0.01, 10.01);
    }

    void show(const KinfuTracker &kinfu) {
        kinfu.getLastFrameCloud(cloud_device_);

        int c;
        cloud_device_.download(cloud_ptr_->points, c);
        cloud_ptr_->width    = cloud_device_.cols();
        cloud_ptr_->height   = cloud_device_.rows();
        cloud_ptr_->is_dense = false;

        cloud_viewer_.removeAllPointClouds();
        cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
        cloud_viewer_.spinOnce();
    }

    PointCloud<PointXYZ>::Ptr    cloud_ptr_;
    DeviceArray2D<PointXYZ>      cloud_device_;
    visualization::PCLVisualizer cloud_viewer_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView {
    KinfuTracker::View                  view_device_;
    KinfuTracker::View                  colors_device_;
    std::vector<KinfuTracker::PixelRGB> view_host_;
    RayCaster::Ptr                      raycaster_ptr_;
    KinfuTracker::DepthMap              generated_depth_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView {
    enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

    SceneCloudView() : extraction_mode_(GPU_Connected6), compute_normals_(false), valid_combined_(false) {
        cloud_ptr_        = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);
        normals_ptr_      = PointCloud<Normal>::Ptr(new PointCloud<Normal>);
        combined_ptr_     = PointCloud<PointNormal>::Ptr(new PointCloud<PointNormal>);
        point_colors_ptr_ = PointCloud<RGB>::Ptr(new PointCloud<RGB>);
    }

    void show(KinfuTracker &kinfu, bool integrate_colors) {
        ScopeTimeT time("PointCloud Extraction");
        cout << "\nGetting cloud... " << flush;

        valid_combined_ = false;

        if (extraction_mode_ != GPU_Connected6) // So use CPU
        {
            kinfu.volume().fetchCloudHost(*cloud_ptr_, extraction_mode_ == CPU_Connected26);
        } else {
            DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud(cloud_buffer_device_);

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
        cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;
    }

    void toggleExtractionMode() {
        extraction_mode_ = (extraction_mode_ + 1) % 3;

        switch (extraction_mode_) {
        case 0:
            cout << "Cloud exctraction mode: GPU, Connected-6" << endl;
            break;
        case 1:
            cout << "Cloud exctraction mode: CPU, Connected-6    (requires a lot of memory)" << endl;
            break;
        case 2:
            cout << "Cloud exctraction mode: CPU, Connected-26   (requires a lot of memory)" << endl;
            break;
        };
    }

    void toggleNormals() {
        compute_normals_ = !compute_normals_;
        cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
    }

    void clearClouds(bool print_message = false) {

        cloud_ptr_->points.clear();
        normals_ptr_->points.clear();
        if (print_message)
            cout << "Clouds/Meshes were cleared" << endl;
    }

    void showMesh(KinfuTracker &kinfu, bool /*integrate_colors*/) {

        ScopeTimeT time("Mesh Extraction");
        cout << "\nGetting mesh... " << flush;

        if (!marching_cubes_)
            marching_cubes_ = MarchingCubes::Ptr(new MarchingCubes());

        DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);
        mesh_ptr_                              = convertToMesh(triangles_device);

        cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
    }

    int  extraction_mode_;
    bool compute_normals_;
    bool valid_combined_;

    PointCloud<PointXYZ>::Ptr cloud_ptr_;
    PointCloud<Normal>::Ptr   normals_ptr_;

    DeviceArray<PointXYZ> cloud_buffer_device_;
    DeviceArray<Normal>   normals_device_;

    PointCloud<PointNormal>::Ptr combined_ptr_;
    DeviceArray<PointNormal>     combined_device_;

    DeviceArray<RGB>     point_colors_device_;
    PointCloud<RGB>::Ptr point_colors_ptr_;

    MarchingCubes::Ptr    marching_cubes_;
    DeviceArray<PointXYZ> triangles_buffer_device_;

    boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuApp {
    enum { PLY = 3, MESH_PLY = 7 };

    KinFuApp(float vsz, int icp, int viz)
        : exit_(false), scan_(false), scan_mesh_(false), scan_volume_(false), independent_camera_(false), registration_(false),
          integrate_colors_(false), focal_length_(-1.f), time_ms_(0), icp_(icp), viz_(viz) {
        // Init Kinfu Tracker
        Eigen::Vector3f volume_size = Vector3f::Constant(vsz /*meters*/);
        kinfu_.volume().setSize(volume_size);

        Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
        Eigen::Vector3f t = volume_size * 0.5f - Vector3f(0, 0, volume_size(2) / 2 * 1.2f);

        Eigen::Affine3f pose = Eigen::Translation3f(t) * Eigen::AngleAxisf(R);

        kinfu_.setInitalCameraPose(pose);
        kinfu_.volume().setTsdfTruncDist(0.030f /*meters*/);
        kinfu_.setIcpCorespFilteringParams(0.1f /*meters*/, sin(pcl::deg2rad(20.f)));
        // kinfu_.setDepthTruncationForICP(5.f/*meters*/);
        kinfu_.setCameraMovementThreshold(0.001f);

        if (!icp)
            kinfu_.disableIcp();

        // Init KinfuApp
        tsdf_cloud_ptr_            = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        image_view_.raycaster_ptr_ = RayCaster::Ptr(new RayCaster(kinfu_.rows(), kinfu_.cols()));
    }

    ~KinFuApp() {
        if (evaluation_ptr_)
            evaluation_ptr_->saveAllPoses(kinfu_);
    }

    void initRegistration() { registration_ = true; }

    void setDepthIntrinsics(std::vector<float> depth_intrinsics) {
        float fx = depth_intrinsics[0];
        float fy = depth_intrinsics[1];

        if (depth_intrinsics.size() == 4) {
            float cx = depth_intrinsics[2];
            float cy = depth_intrinsics[3];
            kinfu_.setDepthIntrinsics(fx, fy, cx, cy);
            cout << "Depth intrinsics changed to fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << endl;
        } else {
            kinfu_.setDepthIntrinsics(fx, fy);
            cout << "Depth intrinsics changed to fx=" << fx << " fy=" << fy << endl;
        }
    }

    void toggleColorIntegration() {
        if (registration_) {
            const int max_color_integration_weight = 2;
            kinfu_.initColorIntegration(max_color_integration_weight);
            integrate_colors_ = true;
        }
        cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
    }

    void toggleEvaluationMode(const string &eval_folder, const string &match_file = string()) {
        evaluation_ptr_ = Evaluation::Ptr(new Evaluation(eval_folder));
        if (!match_file.empty())
            evaluation_ptr_->setMatchFile(match_file);

        kinfu_.setDepthIntrinsics(evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
        image_view_.raycaster_ptr_ = RayCaster::Ptr(new RayCaster(kinfu_.rows(), kinfu_.cols(), evaluation_ptr_->fx, evaluation_ptr_->fy,
                                                                  evaluation_ptr_->cx, evaluation_ptr_->cy));
    }

    void execute(const PtrStepSz<const unsigned short> &depth, const PtrStepSz<const KinfuTracker::PixelRGB> &rgb24, bool has_data) {}

    std::unique_ptr<rs2::context>  ctx;
    std::unique_ptr<rs2::device>   device;
    std::unique_ptr<rs2::pipeline> pipe;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void startMainLoop(bool triggered_capture) {
        rs2::align align_to_depth(RS2_STREAM_DEPTH);
        rs2::align align_to_color(RS2_STREAM_COLOR);

        this->ctx = std::unique_ptr<rs2::context>(new rs2::context());

        rs2::device_list availableDevices = ctx->query_devices();

        printf("There are %d connected RealSense devices.\n", availableDevices.size());
        if (availableDevices.size() == 0) {
            ctx.reset();
            return;
        }

        this->device = std::unique_ptr<rs2::device>(new rs2::device(availableDevices.front()));

        this->pipe = std::unique_ptr<rs2::pipeline>(new rs2::pipeline(*ctx));

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

        // - intrinsics
        rs2_intrinsics intrinsics_depth = depth_stream_profile.get_intrinsics();
        rs2_intrinsics intrinsics_rgb   = color_stream_profile.get_intrinsics();
        rs2_extrinsics rs_extrinsics    = color_stream_profile.get_extrinsics_to(depth_stream_profile);
        kinfu_.setDepthIntrinsics(intrinsics_depth.fx, intrinsics_depth.fy, intrinsics_depth.ppx, intrinsics_depth.ppy);

        {
            std::unique_lock<std::mutex> lock(data_ready_mutex_);

            int currentIndex = 0;

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
                memcpy(const_cast<KinfuTracker::PixelRGB *>(rgb24_.data), rgb.data,
                       color.get_width() * color.get_height() * sizeof(KinfuTracker::PixelRGB));

                execute(depth_, rgb24_, true);
                currentIndex += 1;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void writeCloud(int format) const {
        const SceneCloudView &view = scene_cloud_view_;

        // Points to export are either in cloud_ptr_ or combined_ptr_.
        // If none have points, we have nothing to export.
        if (view.cloud_ptr_->points.empty() && view.combined_ptr_->points.empty()) {
            cout << "Not writing cloud: Cloud is empty" << endl;
        } else {
            if (view.point_colors_ptr_->points.empty()) // no colors
            {
                if (view.valid_combined_)
                    writeCloudFile(format, view.combined_ptr_);
                else
                    writeCloudFile(format, view.cloud_ptr_);
            } else {
                if (view.valid_combined_)
                    writeCloudFile(format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
                else
                    writeCloudFile(format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void writeMesh(int format) const {
        if (scene_cloud_view_.mesh_ptr_)
            writePolygonMeshFile(format, *scene_cloud_view_.mesh_ptr_);
    }

    bool exit_;
    bool scan_;
    bool scan_mesh_;
    bool scan_volume_;

    bool independent_camera_;

    bool  registration_;
    bool  integrate_colors_;
    float focal_length_;

    KinfuTracker kinfu_;

    SceneCloudView scene_cloud_view_;
    ImageView      image_view_;

    KinfuTracker::DepthMap depth_device_;

    pcl::TSDFVolume<float, short>        tsdf_volume_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

    Evaluation::Ptr evaluation_ptr_;

    std::mutex data_ready_mutex_;

    std::vector<KinfuTracker::PixelRGB>     source_image_data_;
    std::vector<unsigned short>             source_depth_data_;
    PtrStepSz<const unsigned short>         depth_;
    PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;

    int time_ms_;
    int icp_, viz_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename CloudPtr> void writeCloudFile(int format, const CloudPtr &cloud_prt) {
    cout << "Saving point cloud to 'cloud.ply' (ASCII)... " << flush;
    pcl::io::savePLYFileASCII("cloud.ply", *cloud_prt);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void writePolygonMeshFile(int format, const pcl::PolygonMesh &mesh) {
    cout << "Saving mesh to to 'mesh.ply'... " << flush;
    pcl::io::savePLYFile("mesh.ply", mesh);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {

    int device_id = 0;
    pcl::gpu::setDevice(device_id);
    pcl::gpu::printShortCudaDeviceInfo(device_id);

    bool triggered_capture = false;

    float volume_size = 3.f;

    int                icp = 1, visualization = 1;
    std::vector<float> depth_intrinsics;

    KinFuApp app(volume_size, icp, visualization);

    app.toggleEvaluationMode("/home/sliu/data/fusion/rgbd_dataset_freiburg1_xyz/", "matches.txt");
    app.initRegistration();
    app.toggleColorIntegration();

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

    KinfuTracker kinfu_;

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
    image_view_.raycaster_ptr_ = RayCaster::Ptr(
        new RayCaster(kinfu_.rows(), kinfu_.cols(), evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy));

    KinfuTracker::DepthMap                  depth_device_;
    PtrStepSz<const unsigned short>         depth_;
    PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;

    SceneCloudView                      scene_cloud_view_;
    KinfuTracker::View                  view_device_;
    KinfuTracker::View                  colors_device_;
    std::vector<KinfuTracker::PixelRGB> view_host_;

    RayCaster::Ptr raycaster_ptr_;

    KinfuTracker::DepthMap generated_depth_;
    while (evaluation_ptr_->grab(currentIndex, depth_, rgb24_)) {
        rs2::frameset frames = pipe->wait_for_frames();
        // frames = align_to_depth.process(frames);
        frames = align_to_color.process(frames);

        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();

        cv::Mat rgba(color.get_height(), color.get_width(), CV_8UC4, (void *)color.get_data());
        cv::Mat rgb;
        cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2BGR);

        memcpy(const_cast<unsigned short *>(depth_.data), depth.get_data(),
               depth.get_width() * depth.get_height() * sizeof(unsigned short));
        memcpy(const_cast<KinfuTracker::PixelRGB *>(rgb24_.data), rgb.data,
               color.get_width() * color.get_height() * sizeof(KinfuTracker::PixelRGB));

        {
            const PtrStepSz<const unsigned short> &        depth = depth_;
            const PtrStepSz<const KinfuTracker::PixelRGB> &rgb24 = rgb24_;
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
            cv::imshow("rgb_image", rgbb);
            char key = cv::waitKey(10);
            if (key == 'q') {
                break;
            }
        }

        currentIndex += 1;
    }
    scene_cloud_view_.show(kinfu_, true);
    const SceneCloudView &view = scene_cloud_view_;
    if (view.point_colors_ptr_->points.empty()) // no colors
    {
        if (view.valid_combined_)
            writeCloudFile(KinFuApp::PLY, view.combined_ptr_);
        else
            writeCloudFile(KinFuApp::PLY, view.cloud_ptr_);
    } else {
        if (view.valid_combined_)
            writeCloudFile(KinFuApp::PLY, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
        else
            writeCloudFile(KinFuApp::PLY, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
    }

    return 0;
}
