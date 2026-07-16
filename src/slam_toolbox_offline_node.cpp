/*
 * slam_toolbox
 * Copyright Work Modifications (c) 2018, Simbe Robotics, Inc.
 * Copyright Work Modifications (c) 2019, Steve Macenski
 *
 * THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
 * COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
 * COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS
 * AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
 *
 * BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
 * BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
 * CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
 * CONDITIONS.
 *
 */

#include <memory>
#include "slam_toolbox/slam_toolbox_offline.hpp"

// Batch entry point: drive the lifecycle transitions ourselves, chew
// through the bag synchronously, save, exit. No executor spin is needed -
// all inputs come from the bag, not from subscriptions.
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<slam_toolbox::OfflineSlamToolbox>(options);

  // a failed transition leaves members like pose_helper_ null - abort with
  // a clean exit code instead of segfaulting on the first scan
  if (node->configure().id() !=
    lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE)
  {
    RCLCPP_ERROR(node->get_logger(), "offline: configure failed.");
    rclcpp::shutdown();
    return 1;
  }
  if (node->activate().id() !=
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
  {
    RCLCPP_ERROR(node->get_logger(), "offline: activate failed.");
    rclcpp::shutdown();
    return 1;
  }

  const bool ok = node->processBag() && node->finishAndSave();

  node->deactivate();
  rclcpp::shutdown();
  return ok ? 0 : 1;
}
