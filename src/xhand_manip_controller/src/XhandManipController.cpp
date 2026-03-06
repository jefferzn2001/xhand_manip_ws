#include "xhand_manip_controller/XhandManipController.h"

namespace legged {
bool XhandManipController::parserObservation(const std::string& name) {
  if (OnnxController::parserObservation(name)) {
    return true;
  }
  // Future: add XHand-specific observation terms here
  // e.g. fingertip keypoints, object pose, contact forces
  return false;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::XhandManipController, controller_interface::ControllerInterface)
