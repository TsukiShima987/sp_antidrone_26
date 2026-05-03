#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{

HikRobot::HikRobot(double exposure_ms, double gain, const std::string & vid_pid, int rotation_angle)
: exposure_us_(exposure_ms * 1e3),
  gain_(gain),
  queue_(1),
  daemon_quit_(false),
  vid_(-1),
  pid_(-1),
  rotation_angle_(rotation_angle)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::HikRobot(
  double exposure_ms, double gain, const std::string & vid_pid, const std::string & serial_number,
  int rotation_angle)
: exposure_us_(exposure_ms * 1e3),
  gain_(gain),
  queue_(1),
  daemon_quit_(false),
  vid_(-1),
  pid_(-1),
  rotation_angle_(rotation_angle),
  serial_number_(serial_number)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
};

HikRobot::~HikRobot()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  tools::logger()->info("HikRobot destructed.");
}

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);
  rotate(data.img, img);

  timestamp = data.timestamp;
}

void HikRobot::capture_start()
{
  capturing_ = false;
  capture_quit_ = false;

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_EnumDevices failed: {:#x}", ret);
    return;
  }

  if (device_list.nDeviceNum == 0) {
    tools::logger()->warn("Not found camera!");
    return;
  }

  int device_index = -1;
  for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
    MV_CC_DEVICE_INFO * pDeviceInfo = device_list.pDeviceInfo[i];
    if (NULL == pDeviceInfo) continue;

    // 检查是否为 USB 设备
    if (pDeviceInfo->nTLayerType == MV_USB_DEVICE) {
      // 获取当前设备的序列号

      if (!serial_number_.empty()) {
        std::string current_sn = (char *)pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber;

        if (current_sn == serial_number_) {
          device_index = i;
          tools::logger()->info("Found camera with Serial Number: {}", serial_number_);
          break;
        }
      } else {
        device_index = 0;
      }
    }
  }

  if (device_index == -1) {
    tools::logger()->warn("Camera with Serial Number {} not found!", serial_number_);
    return;
  }

  // 使用找到的索引创建句柄
  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[device_index]);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CreateHandle failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_OpenDevice failed: {:#x}", ret);
    return;
  }

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, 150);

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StartGrabbing failed: {:#x}", ret);
    return;
  }

  capture_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's capture thread started.");

    capturing_ = true;

    MV_FRAME_OUT raw;
    MV_CC_PIXEL_CONVERT_PARAM cvt_param;

    while (!capture_quit_) {
      std::this_thread::sleep_for(1ms);

      unsigned int ret;
      unsigned int nMsec = 100;

      ret = MV_CC_GetImageBuffer(handle_, &raw, nMsec);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_GetImageBuffer failed: {:#x}", ret);
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      cvt_param.nWidth = raw.stFrameInfo.nWidth;
      cvt_param.nHeight = raw.stFrameInfo.nHeight;

      cvt_param.pSrcData = raw.pBufAddr;
      cvt_param.nSrcDataLen = raw.stFrameInfo.nFrameLen;
      cvt_param.enSrcPixelType = raw.stFrameInfo.enPixelType;

      cvt_param.pDstBuffer = img.data;
      cvt_param.nDstBufferSize = img.total() * img.elemSize();
      cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

      // ret = MV_CC_ConvertPixelType(handle_, &cvt_param);
      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      cv::cvtColor(img, dst_image, type_map.at(pixel_type));
      img = dst_image;

      queue_.push({img, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        break;
      }
    }

    capturing_ = false;
    tools::logger()->info("HikRobot's capture thread stopped.");
  }};
}

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", ret);
    return;
  }
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  if (serial_number_.empty()) {
    // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
    auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
    if (!handle) {
      tools::logger()->warn("Unable to open usb!");
      return;
    }

    if (libusb_reset_device(handle))
      tools::logger()->warn("Unable to reset usb!");
    else
      tools::logger()->info("Reset usb successfully :)");

    libusb_close(handle);
    return;
  }

  libusb_device ** devs;
  ssize_t cnt = libusb_get_device_list(NULL, &devs);
  if (cnt < 0) {
    tools::logger()->warn("Unable to get usb device list!");
    return;
  }

  libusb_device_handle * handle = nullptr;
  for (ssize_t i = 0; i < cnt; i++) {
    libusb_device * dev = devs[i];
    struct libusb_device_descriptor desc;
    if (libusb_get_device_descriptor(dev, &desc) < 0) continue;

    if (desc.idVendor == vid_ && desc.idProduct == pid_) {
      libusb_device_handle * temp_handle = nullptr;
      if (libusb_open(dev, &temp_handle) < 0) continue;

      unsigned char sn_data[256];
      if (
        libusb_get_string_descriptor_ascii(
          temp_handle, desc.iSerialNumber, sn_data, sizeof(sn_data)) > 0) {
        if (std::string((char *)sn_data) == serial_number_) {
          handle = temp_handle;
          break;
        }
      }
      libusb_close(temp_handle);
    }
  }

  libusb_free_device_list(devs, 1);

  if (!handle) {
    tools::logger()->warn("Unable to find usb device with serial number: {}", serial_number_);
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset usb!");
  else
    tools::logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

void HikRobot::rotate(const cv::Mat & src, cv::Mat & dst) const
{
  if (rotation_angle_ == 0) {
    dst = src;
  } else if (rotation_angle_ == 90) {
    cv::rotate(src, dst, cv::ROTATE_90_CLOCKWISE);
  } else if (rotation_angle_ == 180) {
    cv::rotate(src, dst, cv::ROTATE_180);
  } else {
    tools::logger()->warn("Invalid rotation rotation_angle_: {}, using 0", rotation_angle_);
    dst = src;
  }
}

}  // namespace io