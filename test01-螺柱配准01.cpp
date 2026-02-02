//#include <iostream>
//#include <pcl/io/pcd_io.h>                  // PCD文件读写
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/point_types.h>                // 点云类型
//#include <pcl/filters/voxel_grid.h>         // 降采样滤波器
//#include <pcl/features/normal_3d.h>
//#include <pcl/features/normal_3d_omp.h>  // 法向量计算
//#include <pcl/registration/icp.h>
//#include <pcl/registration/icp_nl.h>        // 点到面ICP算法
//#include <pcl/visualization/pcl_visualizer.h> // 可视化
//#include <pcl/filters/approximate_voxel_grid.h> // 体素中心点滤波
//#include <boost/thread/thread.hpp>
//#include <Eigen/Dense>
//#include <pcl/registration/correspondence_estimation.h>
//#include <pcl/filters/crop_box.h> //点云裁剪
//
//#include <pcl/sample_consensus/ransac.h>  //圆柱拟合
//#include <pcl/sample_consensus/sac_model_cylinder.h>
//#include <pcl/segmentation/sac_segmentation.h>
//#include <pcl/sample_consensus/method_types.h>
//#include <pcl/sample_consensus/model_types.h>
//#include <pcl/ModelCoefficients.h>
//#include <pcl/filters/extract_indices.h>
//// 放到你的 #include 列表中，拟合表示
//#include <vtkLineSource.h>
//#include <vtkTubeFilter.h>
// #include <vtkCylinderSource.h>
// #include <vtkTransform.h>
// #include <vtkTransformPolyDataFilter.h>
// #include <vtkPolyDataNormals.h>
// #include <vtkProperty.h>
// #include <vtkLightKit.h>
// #include <vtkMatrix4x4.h>
// #include <vtkActor.h>
//#include <cstdlib>   // for _putenv
//
//typedef pcl::PointXYZ PointT;
//typedef pcl::PointCloud<PointT> PointCloudT;
//
//int main() {
//    _putenv("VTK_USE_LEGACY_OPENGL=1");
//    // 固定参数设置
//    constexpr float  VOXEL_LEAF_SIZE = 1.0f;
//    constexpr int NORMAL_K_SEARCH = 100;
//    constexpr double ICP_MAX_CORRESPONDENCE = 15.0;
//    constexpr int ICP_MAX_ITERATIONS = 300;
//    constexpr double ICP_TRANSFORMATION_EPS = 1e-8;
//
//    // 固定文件路径，请替换为实际路径
//    const std::string sourceCloudPath = "G:/camare_test/test-s10-1.pcd";
//    const std::string targetCloudPath = "G:/camare_test/diban-01-22w.pcd";
//
//    // 1. 加载源点云
//    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZ>);
//    if (pcl::io::loadPCDFile<pcl::PointXYZ>(sourceCloudPath, *sourceCloud) == -1) {
//
//        throw std::runtime_error("无法加载源点云: " + sourceCloudPath);
//    }
//
//    // 2. 加载目标点云
//    pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud(new pcl::PointCloud<pcl::PointXYZ>);
//    if (pcl::io::loadPCDFile<pcl::PointXYZ>(targetCloudPath, *targetCloud) == -1) {
//        throw std::runtime_error("无法加载目标点云: " + targetCloudPath);
//    }
//
//    // 3. 应用体素滤波
//    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredSource(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::VoxelGrid<pcl::PointXYZ> voxelFilter;
//    voxelFilter.setInputCloud(sourceCloud);
//    voxelFilter.setLeafSize(VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE, VOXEL_LEAF_SIZE);
//    voxelFilter.filter(*filteredSource);
//    cout << "滤波后点的个数为：" << filteredSource->size() << endl;
//
//    // 4. 应用初始变换
//    Eigen::Matrix4d initialTransform = Eigen::Matrix4d::Identity();
//    initialTransform << -0.034819818424, 0.404596927334, 0.913831990377, -1600.438384687777,
//        -0.999382554565, -0.018396595727, -0.029934510139, 15.045000611654,
//        0.004699986866, -0.914310063194, 0.404987676931, -168.376711265584,
//        0.0, 0.0, 0.0, 1.0;
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedSource(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedSource01(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::transformPointCloud(*sourceCloud, *transformedSource01, initialTransform);
//    pcl::transformPointCloud(*filteredSource, *transformedSource, initialTransform);
//
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer0"));
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(transformedSource, 0, 255, 0); // green
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(targetCloud, 255, 0, 0);
//
//    viewer->addPointCloud<pcl::PointXYZ>(transformedSource, single_color, "sample cloud");
//    viewer->addPointCloud<pcl::PointXYZ>(targetCloud, target_color, "sampl cloud");
//
//
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }
//
//
//    // 5. 计算法线
//    pcl::PointCloud<pcl::PointNormal>::Ptr sourceWithNormals(new pcl::PointCloud<pcl::PointNormal>);
//    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
//    normalEstimation.setInputCloud(transformedSource);
//    normalEstimation.setKSearch(NORMAL_K_SEARCH);
//    normalEstimation.setNumberOfThreads(4);
//    pcl::PointCloud<pcl::Normal>::Ptr sourceNormals(new pcl::PointCloud<pcl::Normal>);
//    normalEstimation.compute(*sourceNormals);
//    pcl::concatenateFields(*transformedSource, *sourceNormals, *sourceWithNormals);
//
//    pcl::PointCloud<pcl::PointNormal>::Ptr targetWithNormals(new pcl::PointCloud<pcl::PointNormal>);
//    normalEstimation.setInputCloud(targetCloud);
//    pcl::PointCloud<pcl::Normal>::Ptr targetNormals(new pcl::PointCloud<pcl::Normal>);
//    normalEstimation.compute(*targetNormals);
//    pcl::concatenateFields(*targetCloud, *targetNormals, *targetWithNormals);
//    cout << "法线计算完成" <<  endl;
//
//    // 6. ICP配准
//
//    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
//    icp.setInputSource(sourceWithNormals);
//    icp.setInputTarget(targetWithNormals);
//    icp.setMaxCorrespondenceDistance(ICP_MAX_CORRESPONDENCE);
//    icp.setMaximumIterations(ICP_MAX_ITERATIONS);
//    icp.setTransformationEpsilon(ICP_TRANSFORMATION_EPS);
//
//    pcl::PointCloud<pcl::PointNormal>::Ptr alignedCloud(new pcl::PointCloud<pcl::PointNormal>);
//    icp.align(*alignedCloud);
//    cout << "配准完成"  << endl;
//
//    double fitness_score = icp.getFitnessScore();
//    std::cout << "配准误差: " << fitness_score << std::endl;
//
//    if (!icp.hasConverged()) {
//
//        throw std::runtime_error("ICP配准未收敛");
//    };
//    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::transformPointCloud(*transformedSource01, *out_cloud, icp.getFinalTransformation());
//    std::cout << "Final transformation matrix:\n" << icp.getFinalTransformation() << std::endl;
//
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_1(new pcl::visualization::PCLVisualizer("3D Viewer"));
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(out_cloud, 0, 255, 0); // green
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color1(targetCloud, 255, 0, 0);
//
//    viewer_1->addPointCloud<pcl::PointXYZ>(out_cloud, single_color1, "sample-1 cloud");
//    viewer_1->addPointCloud<pcl::PointXYZ>(targetCloud, target_color1, "sampl-1 cloud");
//
//
//    while (!viewer_1->wasStopped())
//    {
//        viewer_1->spinOnce(100);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }
//
//    Eigen::Matrix4d T_hand_eye = Eigen::Matrix4d::Identity();
//    T_hand_eye << -0.708389, -0.0115623, -0.705728, -30.2151,
//        0.705738, 0.00391852, -0.708463, -169.905,
//        0.0109569, -0.999925, 0.0053841, 94.782,
//        0.0, 0.0, 0.0, 1.0;
//    Eigen::Matrix4d T_icp =  icp.getFinalTransformation().cast<double>() * initialTransform;
//    Eigen::Matrix4d T_final = T_icp.inverse();
//    Eigen::Matrix3d rotation = T_final.block<3, 3>(0, 0);
//    Eigen::Vector4d P_base(-842.5, 0, 67.5, 1.0);
//    Eigen::Vector3d normal_base(0, 0, 1.0);
//    Eigen::Vector4d P_flange = T_final * P_base;
//    Eigen::Vector3d  P_flange3d = P_flange.head<3>();
//    Eigen::Vector3d normal_flange = rotation * normal_base;
//    normal_flange.normalize(); // 确保法向量长度为1
//
//    std::cout << P_flange3d << std::endl;
//    std::cout << "基准螺柱的轴线向量" << std::endl;
//    std::cout << normal_flange << std::endl;
//
//    //点云融合
//    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merged(new pcl::PointCloud<pcl::PointXYZ>);
//    //*cloud_merged = *cloud_target; // 添加目标点云
//    //cloud_merged->insert(cloud_merged->end(), out_cloud->points.begin(), out_cloud->points.end()); // 添加配准后的源点云
//
//    // 4. 根据目标点云的坐标系裁剪点云
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped_source(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cropped_target(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::CropBox<pcl::PointXYZ> crop_box_filter;
//
//    // 设置裁剪区域（根据目标点云的坐标系）
//    // 假设裁剪区域为一个立方体，范围为 x: [-1.0, 1.0], y: [-1.0, 1.0], z: [-1.0, 1.0]
//    Eigen::Vector4f min_point(-846.5, -6.0, 62, 1.0); // 最小点 (x, y, z, 1)
//    Eigen::Vector4f max_point(-838.5, 6.0, 85.0, 1.0);   // 最大点 (x, y, z, 1)
//
//
//    crop_box_filter.setMin(min_point);
//    crop_box_filter.setMax(max_point);
//    crop_box_filter.setInputCloud(targetCloud);
//    crop_box_filter.filter(*cloud_cropped_target);
//
//    crop_box_filter.setInputCloud(out_cloud);
//    crop_box_filter.filter(*cloud_cropped_source);
//    cout << "滤波后点的个数为：" << cloud_cropped_source->size() << endl;
//
//    pcl::io::savePCDFile("G:/camare_test/cloud_cropped_source-s04.pcd", *cloud_cropped_source);
//
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_2(new pcl::visualization::PCLVisualizer("3D Viewer2"));
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(cloud_cropped_target, 255, 0, 0); // green
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color2(cloud_cropped_source, 0, 255, 0); //red
//    viewer_2->addPointCloud<pcl::PointXYZ>(cloud_cropped_source, target_color2, "sample2 cloud");
//    viewer_2->addPointCloud<pcl::PointXYZ>(cloud_cropped_target, single_color2, "sample-2 cloud");
//
//    while (!viewer_2->wasStopped())
//    {
//        viewer_2->spinOnce(100);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }
//
//    //1. 加载源点云和目标点云
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
//
//    if (pcl::io::loadPCDFile<pcl::PointXYZ>("G:/camare_test/cloud_cropped_source-s04.pcd", *cloud_source) == -1) {
//        PCL_ERROR("Couldn't read file source_cloud.pcd \n");
//        return (-1);
//
//    }
//
//    pcl::NormalEstimation<PointT, pcl::Normal> ne;
//    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
//    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
//    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//    pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
//    pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
//
//    ne.setSearchMethod(tree);
//    ne.setInputCloud(cloud_source);
//    ne.setKSearch(40);
//    ne.compute(*cloud_normals);
//
//    seg.setOptimizeCoefficients(true);
//    seg.setModelType(pcl::SACMODEL_CYLINDER);
//    seg.setMethodType(pcl::SAC_RANSAC);
//    Eigen::Vector3f known_axis(0, 0, 1); // 已知轴线方向（例如Z轴方向）
//    seg.setAxis(known_axis);             // 设置轴线约束
//    seg.setEpsAngle(0.2);
//    seg.setNormalDistanceWeight(0.05);//法向量影响比重
//    seg.setMaxIterations(10000);
//    seg.setDistanceThreshold(0.35);//点到圆柱面的最大距离阈值。所有距离小于该值的点会被视为当前圆柱模型的“内点”（inliers）。
//    seg.setRadiusLimits(2.7, 3.3); // 螺柱半径为3mm
//    seg.setInputCloud(cloud_source);
//    seg.setInputNormals(cloud_normals);
//    seg.segment(*inliers_cylinder, *coefficients_cylinder);
//
//    if (inliers_cylinder->indices.empty()) {
//        std::cerr << "无法找到合适的圆柱模型！" << std::endl;
//        return -1;
//    }
//
//
//    // 提取拟合的圆柱点云
//    pcl::ExtractIndices<PointT> extract;
//    extract.setInputCloud(cloud_source);
//    extract.setIndices(inliers_cylinder);
//    extract.setNegative(false);
//    PointCloudT::Ptr cloud_cylinder(new PointCloudT);
//    extract.filter(*cloud_cylinder);
//    std::cout << "轴线上随机点: (" << coefficients_cylinder->values[0] << ", "
//        << coefficients_cylinder->values[1] << ", "
//        << coefficients_cylinder->values[2] << ")" << std::endl;
//    std::cout << "Cylinder coefficients: " << std::endl;
//    std::cout << "  Radius: " << coefficients_cylinder->values[6] << std::endl;
//
//
//    Eigen::Vector3f point_on_axis(coefficients_cylinder->values[0], coefficients_cylinder->values[1], coefficients_cylinder->values[2]);
//
//    // 获取圆柱轴线的方向向量
//    Eigen::Vector3f axis(coefficients_cylinder->values[3], coefficients_cylinder->values[4], coefficients_cylinder->values[5]);
//    axis.normalize(); // 归一化
//    Eigen::Vector3f z_neg_direction(0, 0, 1);
//    if (axis.dot(z_neg_direction) < 0) {
//        axis = -axis; // 如果当前轴线方向与Z轴负方向夹角大于90度，则反转方向
//        std::cout << "轴线方向已调整为指向Z轴正方向" << std::endl;
//    }
//    std::cout << "  axis: " << axis << std::endl;
//   
//    // 计算圆柱的长度
//    float min_proj = std::numeric_limits<float>::max();
//    float max_proj = -std::numeric_limits<float>::max();
//
//    for (const auto& point : cloud_cylinder->points) {
//        float proj = (point.x - point_on_axis[0]) * axis[0] +
//            (point.y - point_on_axis[1]) * axis[1] +
//            (point.z - point_on_axis[2]) * axis[2]; // 点投影到轴线上
//        if (proj < min_proj) min_proj = proj;
//        if (proj > max_proj) max_proj = proj;
//    }
//
//    float cylinder_length = max_proj - min_proj;
//    std::cout << "  Cylinder length: " << cylinder_length << std::endl;
//
//    // 计算 min_proj 对应的实际坐标
//    Eigen::Vector3f min_proj_point = point_on_axis + axis * min_proj;
//
//    // 打印 min_proj 对应的实际坐标
//    std::cout << "Min projection point: (" << min_proj_point[0] << ", " << min_proj_point[1] << ", " << min_proj_point[2] << ")" << std::endl;
//
//    Eigen::Vector4f min_proj_point4f(min_proj_point(0), min_proj_point(1), min_proj_point(2), 1.0);
//    Eigen::Vector4d min_proj_point4d=min_proj_point4f.cast<double>();
//    Eigen::Vector4d P_flange_01 = T_final * min_proj_point4d;
//    Eigen::Vector3d P_flange_01_3d = P_flange_01.head<3>();
//    Eigen::Vector3d normal_flange_01 = rotation * axis.cast<double>();
//    normal_flange_01.normalize(); // 确保法向量长度为1
//
//    std::cout << P_flange_01_3d << std::endl;
//    std::cout << "拟合螺柱的轴线向量" << std::endl;
//    std::cout << normal_flange_01 << std::endl;
//
//    float dot_product = z_neg_direction.dot(axis);
//    //dot_product = std::clamp(dot_product, -1.0, 1.0);  // 防止浮点数误差导致 acos 越界
//    float angle_rad = std::acos(dot_product);         // 弧度
//    float angle_deg = angle_rad * 180.0 / M_PI;       // 转换为角度
//    std::cout << "轴线夹角（角度）: " << angle_deg << "°" << std::endl;
//
//
//
//    // ===== 已有：point_on_axis、axis(已normalize)、min_proj、max_proj、radius r =====
//  // 这些值你在拟合后已经算好了：见你当前文件里 axis/投影/长度计算的位置。:contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
//
//  // 端点、中心、长度
//    Eigen::Vector3f c0 = point_on_axis + axis * min_proj;
//    Eigen::Vector3f c1 = point_on_axis + axis * max_proj;
//    // 1) 做一条线段（拟合轴的“实长”）
//    auto line = vtkSmartPointer<vtkLineSource>::New();
//    line->SetPoint1(c0.x(), c0.y(), c0.z());
//    line->SetPoint2(c1.x(), c1.y(), c1.z());
//
//    // 2) 把线段“拉粗”成圆柱（带端盖）
//    auto tube = vtkSmartPointer<vtkTubeFilter>::New();
//    tube->SetInputConnection(line->GetOutputPort());
//    tube->SetRadius(coefficients_cylinder->values[6]);
//    tube->SetNumberOfSides(96);    // 圆滑程度
//    tube->CappingOn();
//    tube->Update();
//
//
//
//    // 4) 加入可视化
//    pcl::visualization::PCLVisualizer viewer03("Cylinder fitting");
//    viewer03.addModelFromPolyData(tube->GetOutput(), "stud_full");
//    viewer03.addPointCloud<PointT>(cloud_source, "filtered_cloud");
//    viewer03.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0     , "filtered_cloud");
//
//    // ====== 4) 提升亮度/可见性（材质 + 灯光）======
//    auto shape_map_ptr = viewer03.getShapeActorMap();
//    if (shape_map_ptr) {
//        auto it = shape_map_ptr->find("stud_full");
//        if (it != shape_map_ptr->end()) {
//            vtkProp* base = it->second.Get();
//            vtkActor* actor = vtkActor::SafeDownCast(base);
//            vtkLODActor* lod = vtkLODActor::SafeDownCast(base);
//            vtkProperty* prop = actor ? actor->GetProperty() :
//                (lod ? lod->GetProperty() : nullptr);
//            if (prop) {
//                // 更亮的金黄 + 半透明
//                prop->SetColor(1.0, 0.95, 0.25);
//                prop->SetOpacity(0.50);            // 0.45~0.65 之间可微调
//                // 明暗与高光（端面/侧壁对比更明显）
//                prop->SetInterpolationToPhong();
//                prop->SetAmbient(0.35);            // 环境光占比（提亮整体）
//                prop->SetDiffuse(0.90);
//                prop->SetSpecular(1.0);            // 高光强度
//                prop->SetSpecularPower(80.0);      // 高光锐度（越大越“亮点”）
//                prop->BackfaceCullingOn();         // 让端面在旋转时反差更清晰
//                // prop->EdgeVisibilityOn(); prop->SetEdgeColor(0,0,0); // 边线（可选）
//            }
//        }
//    }
//
//
//
//
//
//    // 7) 加一套三点布光（侧光/背光），立体感更强（可选）
//    auto ren = viewer03.getRenderWindow()->GetRenderers()->GetFirstRenderer();
//    auto lightKit = vtkSmartPointer<vtkLightKit>::New();
//    lightKit->AddLightsToRenderer(ren);
//
//    
//    while (!viewer03.wasStopped())
//    {
//        viewer03.spinOnce(100);
//    }
//    return 0;
//}