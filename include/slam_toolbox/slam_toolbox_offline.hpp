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

#ifndef SLAM_TOOLBOX__SLAM_TOOLBOX_OFFLINE_HPP_
#define SLAM_TOOLBOX__SLAM_TOOLBOX_OFFLINE_HPP_

#include <deque>
#include <memory>
#include <set>
#include <string>

#include "slam_toolbox/slam_toolbox_common.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

namespace slam_toolbox
{

// Offline mapping: reads scans, TF and pose priors straight out of a rosbag
// via rosbag2_cpp and feeds them through the same karto pipeline as the
// asynchronous node - no DDS transport, no /clock pacing, no TF message
// filter, no queues to overflow. Throughput is bounded by scan matching and
// solver CPU only, and the result is deterministic for a given bag.
class OfflineSlamToolbox : public SlamToolbox
{
public:
  explicit OfflineSlamToolbox(rclcpp::NodeOptions options);
  ~OfflineSlamToolbox() {}

  // Offline sessions must be hermetic: every input comes from the bag, so
  // the live TF listener and the scan/prior subscriptions the base class
  // creates are torn down right after it sets them up - otherwise traffic
  // from a shared DDS domain would silently leak into the map and break
  // the determinism this node exists to provide.
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;

  // Sequentially process the whole bag; returns false if the bag could not
  // be read or no scan was ever processed.
  bool processBag();
  // Final optimization, pose-graph serialization and pgm/yaml export.
  bool finishAndSave();

protected:
  void laserCallback(
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan) override;

  // Feed a bag TF message into the buffer, dropping filtered parent frames
  // (the production map->odom must not leak into the mapping session).
  void ingestTf(const tf2_msgs::msg::TFMessage & msg, bool is_static);
  // When continuing a deserialized graph: pre-scan the bag's pose priors
  // for the current anchor position of every source and shift stale stored
  // priors accordingly (offline replacement for the update_prior_anchors
  // service the online continue flow calls). First message per source wins,
  // matching the online flow's first /layout/cameras snapshot.
  void reanchorFromBagPriors();
  // Process buffered scans whose stamp is at or before up_to. Scans are
  // held back by offline_scan_lag of bag time so the TF interpolation
  // window and the pose priors around each scan are complete, mirroring
  // what queuing latency provides in the online pipeline.
  void drainScans(const rclcpp::Time & up_to);
  bool saveMap(const std::string & basename);

  std::string offline_bag_;
  std::string offline_output_basename_;
  double offline_scan_lag_;
  std::set<std::string> drop_tf_parents_;

  std::deque<sensor_msgs::msg::LaserScan::ConstSharedPtr> pending_scans_;
  size_t scans_seen_{0}, scans_processed_{0}, scans_no_odom_{0};
  size_t priors_ingested_{0}, nodes_added_{0};
};

}  // namespace slam_toolbox

#endif  // SLAM_TOOLBOX__SLAM_TOOLBOX_OFFLINE_HPP_
