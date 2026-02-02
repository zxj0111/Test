//#include <iostream>
//#include <pcl/io/png_io.h>
//
//#include <stdexcept>// PCD文件读写
//
//
//#include <area_scan_3d_camera/Camera.h>
//#include <area_scan_3d_camera/api_util.h>
//#include <area_scan_3d_camera/Frame2D.h>
//#include <area_scan_3d_camera/Frame3D.h>
//#include <area_scan_3d_camera/Frame2DAnd3D.h>
//#include <area_scan_3d_camera/parameters/Scanning2D.h>
//#include <area_scan_3d_camera/parameters/Scanning3D.h>
//
//#include <string>
//#include <vector>
//#include <Eigen/Dense>
//
//
//
//
//static void SaveMechEyeColor2DToPNG_ByPCL_Crop(const mmind::eye::Color2DImage& bgrImg,
//    const Eigen::Vector4i& roi,
//    const std::string& pngPath)
//{
//    const int W = (int)bgrImg.width();
//    const int H = (int)bgrImg.height();
//    const auto* p = bgrImg.data();
//    if (!p || W <= 0 || H <= 0) throw std::runtime_error("Empty Color2DImage.");
//
//    int x = roi[0], y = roi[1], w = roi[2], h = roi[3];
//    if (x < 0) x = 0; if (y < 0) y = 0;
//    if (x + w > W) w = W - x;
//    if (y + h > H) h = H - y;
//    if (w <= 0 || h <= 0) throw std::runtime_error("Invalid crop ROI for 2D image.");
//
//    std::vector<unsigned char> rgb((size_t)w * h * 3);
//    for (int yy = 0; yy < h; ++yy) {
//        for (int xx = 0; xx < w; ++xx) {
//            const auto& c = p[(y + yy) * W + (x + xx)]; // BGR
//            const int i = (yy * w + xx) * 3;
//            rgb[i + 0] = (unsigned char)c.r;
//            rgb[i + 1] = (unsigned char)c.g;
//            rgb[i + 2] = (unsigned char)c.b;
//        }
//    }
//
//    pcl::io::saveRgbPNGFile(pngPath, rgb.data(), w, h);
//}
//
//
//// Mech-Eye SDK headers
//
//
//
//struct CaptureConfig
//{
//    // 3D
//    std::vector<double> exposure3D_ms; // Scan3DExposureSequence, ms :contentReference[oaicite:13]{index=13}
//    mmind::eye::ROI roi3D;             // Scan3DROI :contentReference[oaicite:14]{index=14}
//
//    // 2D（示例：Timed 模式 + 曝光时间；如果你用 Auto/HDR/Flash，自行改 ExposureMode 和相关参数）
//
//    mmind::eye::ROI autoExposureRoi2D{ 0,0,0,0 };
//    // 超时
//    unsigned int captureTimeoutMs = 15000;
//};
//
//struct DualCameras
//{
//    mmind::eye::Camera cam01;
//    mmind::eye::Camera cam02;
//};
//
//// ① 只负责实例化 + 连接两台相机
//static DualCameras ConnectTwoCameras()
//{
//    DualCameras cams;
//
//    mmind::eye::showError(cams.cam01.connect("192.168.1.10", 10000)); // connect(ip, timeoutMs):contentReference[oaicite:16]{index=16}
//    std::cout << "Connected to camera01.\n";
//
//    //mmind::eye::showError(cams.cam02.connect("169.254.246.15", 10000));
//    //std::cout << "Connected to camera02.\n";
//
//    return cams;
//}
//
//// 通用：把“ROI + 曝光”等参数写进当前 UserSet（两台相机都能复用）
//static void Apply2D3DSettings(mmind::eye::Camera& cam, const CaptureConfig& cfg)
//{
//    auto& us = cam.currentUserSet(); // 通过 UserSet 配置采集参数:contentReference[oaicite:17]{index=17}
//
//    // --- 3D 参数 ---
//    mmind::eye::showError(us.setFloatArrayValue(
//        mmind::eye::scanning3d_setting::ExposureSequence::name, // Scan3DExposureSequence:contentReference[oaicite:18]{index=18}
//        cfg.exposure3D_ms));                                    // setFloatArrayValue:contentReference[oaicite:19]{index=19}
//
//    mmind::eye::showError(us.setRoiValue(
//        mmind::eye::scanning3d_setting::ROI::name, // Scan3DROI:contentReference[oaicite:20]{index=20}
//        cfg.roi3D));                               // setRoiValue:contentReference[oaicite:21]{index=21}
//
//    // --- 2D 参数（示例：ExposureMode + ExposureTime）---
//    mmind::eye::showError(us.setEnumValue(mmind::eye::scanning2d_setting::ExposureMode::name, "Auto")); // setEnumValue 支持字符串:contentReference[oaicite:22]{index=22}
//  
// 
//
//    // Auto 模式下你才需要设置 AutoExposureROI（参数名为 Scan2DROI）
//
//    mmind::eye::showError(us.setRoiValue(mmind::eye::scanning2d_setting::AutoExposureROI::name, cfg.autoExposureRoi2D));
//
//
//    
//
//    // 如果你希望把这套参数“写进相机里持久化”，可打开这一句：
//    // showError(us.saveAllParametersToDevice()); :contentReference[oaicite:23]{index=23}
//}
//
//// ② camera01 的 2D+3D 采集函数（含 ROI/曝光设置 + 保存点云）
//static void CaptureCam01_2D3D(mmind::eye::Camera& cam01)
//{
//    CaptureConfig cfg;
//    cfg.exposure3D_ms = { 9.0 };
//    cfg.roi3D = mmind::eye::ROI(512,353,763,399);
//    cfg.autoExposureRoi2D = mmind::eye::ROI(512, 353, 763, 399);
//    const Eigen::Vector4i roi(512, 353, 763, 399);
//
//    cfg.captureTimeoutMs = 15000;
//
//    Apply2D3DSettings(cam01, cfg);
//
//    mmind::eye::Frame2DAnd3D frame; // 同时拿 2D+3D:contentReference[oaicite:24]{index=24}
//    mmind::eye::showError(cam01.capture2DAnd3D(frame, cfg.captureTimeoutMs)); // capture2DAnd3D:contentReference[oaicite:25]{index=25}
//
//   
//
//    //// 保存 3D 点云（无纹理）
//    //const std::string cloudFile = "luozhu-01/test-s10.pcd";
//    
//    // // 相机1点云保存到路径
//    const std::string cloudFile = "G:/camare_test/cam01_cloud.pcd";
//    mmind::eye::showError(frame.frame3D().saveUntexturedPointCloud(mmind::eye::FileFormat::PCD, cloudFile));
//    
//    //// saveUntexturedPointCloud 的语义/organized 选项见文档:contentReference[oaicite:26]{index=26}
//    auto colorImg = frame.frame2D().getColorImage();   
//    // 相机1图片保存到某路径（也可以用正斜杠，更简单）
//    SaveMechEyeColor2DToPNG_ByPCL_Crop(colorImg, roi, "G:/camare_test/cam01_img.png");
//
//    //SaveMechEyeColor2DToPNG_ByPCL_Crop(colorImg, roi,"luozhu-01/cam03.png");
//
//    
//}
////
////// ③ camera02 的 2D+3D 采集函数
////static void CaptureCam02_2D3D(mmind::eye::Camera& cam02)
////{
////    CaptureConfig cfg;
////    cfg.exposure3D_ms = { 16.0 };
////    cfg.roi3D = mmind::eye::ROI(10, 307, 625, 635);
////    cfg.autoExposureRoi2D = mmind::eye::ROI(10, 307, 625, 635);
////    const Eigen::Vector4i roi(10, 307, 625, 635);
////    cfg.captureTimeoutMs = 15000;
////
////    Apply2D3DSettings(cam02, cfg);
////
////    mmind::eye::Frame2DAnd3D frame;
////    mmind::eye::showError(cam02.capture2DAnd3D(frame, cfg.captureTimeoutMs));
////
////    const std::string cloudFile = "luozhu-01/test-s11.pcd";
////    mmind::eye::showError(frame.frame3D().saveUntexturedPointCloud(mmind::eye::FileFormat::PCD, cloudFile));
////
////    auto colorImg = frame.frame2D().getColorImage();
////    SaveMechEyeColor2DToPNG_ByPCL_Crop(colorImg, roi, "luozhu-01/cam02.png");
////}
//
//int main()
//{
//    try {
//        auto cams = ConnectTwoCameras();
//
//        CaptureCam01_2D3D(cams.cam01);
//       /* CaptureCam02_2D3D(cams.cam02);*/
//
//        cams.cam01.disconnect();
//       /* cams.cam02.disconnect();*/
//    }
//    catch (const std::exception& e) {
//        std::cerr << "Error: " << e.what() << "\n";
//        return -1;
//    }
//    return 0;
//}
