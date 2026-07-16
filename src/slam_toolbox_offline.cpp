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

#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_storage/storage_filter.hpp"
#include "slam_toolbox/slam_toolbox_offline.hpp"
#include "slam_toolbox/visualization_utils.hpp"

namespace slam_toolbox
{

/*****************************************************************************/
OfflineSlamToolbox::OfflineSlamToolbox(rclcpp::NodeOptions options)
: SlamToolbox(options)
/*****************************************************************************/
{
  this->declare_parameter("offline_bag", std::string(""));
  this->declare_parameter("offline_output_basename", std::string(""));
  this->declare_parameter("offline_scan_lag", 1.0);
  this->declare_parameter(
    "offline_drop_tf_parent_frames", std::vector<std::string>{"map"});
}

/*****************************************************************************/
CallbackReturn OfflineSlamToolbox::on_configure(
  const rclcpp_lifecycle::State & state)
/*****************************************************************************/
{
  const CallbackReturn result = SlamToolbox::on_configure(state);
  // drop the live TF listener - bag transforms are fed via ingestTf, and
  // live /tf from a shared domain must not leak into the buffer
  tfL_.reset();
  return result;
}

/*****************************************************************************/
CallbackReturn OfflineSlamToolbox::on_activate(
  const rclcpp_lifecycle::State & state)
/*****************************************************************************/
{
  const CallbackReturn result = SlamToolbox::on_activate(state);
  // drop the live scan and prior subscriptions for the same reason
  scan_filter_.reset();
  scan_filter_sub_.reset();
  prior_sub_.reset();
  return result;
}

/*****************************************************************************/
void OfflineSlamToolbox::laserCallback(
  sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
/*****************************************************************************/
{
  // identical to the asynchronous node, minus the transport in front of it
  scan_header = scan->header;

  Pose2 pose;
  if (!pose_helper_->getOdomPose(pose, scan->header.stamp)) {
    scans_no_odom_++;
    if (scans_no_odom_ == 1 || scans_no_odom_ % 100 == 0) {
      RCLCPP_WARN(get_logger(), "offline: failed to compute odom pose for "
        "%zu scans so far (TF gap in the bag?)", scans_no_odom_);
    }
    return;
  }

  LaserRangeFinder * laser = getLaser(scan);
  if (!laser) {
    RCLCPP_WARN(get_logger(), "offline: failed to create laser device for"
      " %s; discarding scan", scan->header.frame_id.c_str());
    return;
  }

  if (shouldProcessScan(scan, pose)) {
    if (addScan(laser, scan, pose)) {
      nodes_added_++;
    }
    scans_processed_++;
  }
}

/*****************************************************************************/
void OfflineSlamToolbox::ingestTf(
  const tf2_msgs::msg::TFMessage & msg, bool is_static)
/*****************************************************************************/
{
  for (const auto & transform : msg.transforms) {
    if (drop_tf_parents_.count(transform.header.frame_id)) {
      continue;
    }
    try {
      tf_->setTransform(transform, "bag_offline", is_static);
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_ONCE(get_logger(), "offline: rejected bag transform "
        "%s->%s: %s", transform.header.frame_id.c_str(),
        transform.child_frame_id.c_str(), e.what());
    }
  }
}

/*****************************************************************************/
void OfflineSlamToolbox::drainScans(const rclcpp::Time & up_to)
/*****************************************************************************/
{
  while (!pending_scans_.empty() &&
    rclcpp::Time(pending_scans_.front()->header.stamp) <= up_to)
  {
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan =
      pending_scans_.front();
    pending_scans_.pop_front();
    laserCallback(scan);
  }
}

/*****************************************************************************/
void OfflineSlamToolbox::reanchorFromBagPriors()
/*****************************************************************************/
{
  {
    boost::mutex::scoped_lock lock(smapper_mutex_);
    if (!smapper_ || !smapper_->getMapper() ||
      !smapper_->getMapper()->GetGraph() ||
      smapper_->getMapper()->GetGraph()->GetVertices().empty())
    {
      return;  // fresh map - nothing to re-anchor
    }
  }

  rosbag2_cpp::Reader reader;
  reader.open(offline_bag_);
  rosbag2_storage::StorageFilter filter;
  filter.topics = {pose_prior_topic_};
  reader.set_filter(filter);

  rclcpp::Serialization<slam_toolbox::msg::PosePrior> prior_serde;
  std::map<std::string, karto::Vector2<kt_double>> anchors;
  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    const rclcpp::SerializedMessage serialized(*bag_msg->serialized_data);
    slam_toolbox::msg::PosePrior prior;
    prior_serde.deserialize_message(&serialized, &prior);
    anchors.emplace(prior.source_id, karto::Vector2<kt_double>(
        prior.anchor_position.x, prior.anchor_position.y));
  }

  if (anchors.empty()) {
    RCLCPP_WARN(get_logger(), "offline: continuing a pose graph but the bag "
      "has no %s messages - stale anchors cannot be updated.",
      pose_prior_topic_.c_str());
    return;
  }

  const int updated = applyPriorAnchorUpdates(anchors);
  RCLCPP_INFO(get_logger(), "offline: re-anchored priors against %zu "
    "sources from the bag (%d nodes shifted).", anchors.size(), updated);
}

/*****************************************************************************/
bool OfflineSlamToolbox::processBag()
/*****************************************************************************/
{
  offline_bag_ = this->get_parameter("offline_bag").as_string();
  offline_output_basename_ =
    this->get_parameter("offline_output_basename").as_string();
  offline_scan_lag_ = this->get_parameter("offline_scan_lag").as_double();
  const auto drop_frames =
    this->get_parameter("offline_drop_tf_parent_frames").as_string_array();
  drop_tf_parents_ = std::set<std::string>(
    drop_frames.begin(), drop_frames.end());

  if (offline_bag_.empty()) {
    RCLCPP_ERROR(get_logger(), "offline: offline_bag parameter is not set.");
    return false;
  }

  try {
    reanchorFromBagPriors();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "offline: prior re-anchoring failed: %s",
      e.what());
    return false;
  }

  rosbag2_cpp::Reader reader;
  try {
    reader.open(offline_bag_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "offline: failed to open bag %s: %s",
      offline_bag_.c_str(), e.what());
    return false;
  }

  rosbag2_storage::StorageFilter filter;
  filter.topics = {scan_topic_, "/tf", "/tf_static", pose_prior_topic_};
  reader.set_filter(filter);

  const auto metadata = reader.get_metadata();
  const int64_t bag_start =
    metadata.starting_time.time_since_epoch().count();
  const int64_t bag_ns = metadata.duration.count();
  RCLCPP_INFO(get_logger(), "offline: processing %s (%.1f min of bag time, "
    "scan lag %.1f s)", offline_bag_.c_str(), bag_ns / 60.0e9,
    offline_scan_lag_);

  rclcpp::Serialization<sensor_msgs::msg::LaserScan> scan_serde;
  rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_serde;
  rclcpp::Serialization<slam_toolbox::msg::PosePrior> prior_serde;
  const rclcpp::Duration lag = rclcpp::Duration::from_seconds(
    offline_scan_lag_);

  const auto wall_start = std::chrono::steady_clock::now();
  int64_t last_report_ns = bag_start;

  while (rclcpp::ok() && reader.has_next()) {
    auto bag_msg = reader.read_next();
    const rclcpp::Time bag_time(bag_msg->recv_timestamp, RCL_ROS_TIME);
    const rclcpp::SerializedMessage serialized(*bag_msg->serialized_data);

    if (bag_msg->topic_name == scan_topic_) {
      auto scan = std::make_shared<sensor_msgs::msg::LaserScan>();
      scan_serde.deserialize_message(&serialized, scan.get());
      pending_scans_.push_back(scan);
      scans_seen_++;
    } else if (bag_msg->topic_name == "/tf_static") {
      tf2_msgs::msg::TFMessage tf_msg;
      tf_serde.deserialize_message(&serialized, &tf_msg);
      ingestTf(tf_msg, true);
    } else if (bag_msg->topic_name == "/tf") {
      tf2_msgs::msg::TFMessage tf_msg;
      tf_serde.deserialize_message(&serialized, &tf_msg);
      ingestTf(tf_msg, false);
    } else if (bag_msg->topic_name == pose_prior_topic_) {
      auto prior = std::make_shared<slam_toolbox::msg::PosePrior>();
      prior_serde.deserialize_message(&serialized, prior.get());
      posePriorCallback(prior);
      priors_ingested_++;
    }

    drainScans(bag_time - lag);

    if (bag_msg->recv_timestamp - last_report_ns > 60'000'000'000) {
      last_report_ns = bag_msg->recv_timestamp;
      const double wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();
      const double done_ns = static_cast<double>(
        bag_msg->recv_timestamp - bag_start);
      RCLCPP_INFO(get_logger(), "offline: %.0f%% of bag (%.1f/%.1f min) in "
        "%.1f min wall (%.1fx real time); %zu/%zu scans processed, %zu "
        "nodes, %d priors attached", 100.0 * done_ns / bag_ns,
        done_ns / 60.0e9, bag_ns / 60.0e9, wall_s / 60.0,
        done_ns / 1.0e9 / wall_s, scans_processed_, scans_seen_,
        nodes_added_, attached_priors_);
    }
  }

  // flush scans still held back by the lag window
  drainScans(rclcpp::Time(std::numeric_limits<int64_t>::max(), RCL_ROS_TIME));

  const double wall_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - wall_start).count();
  RCLCPP_INFO(get_logger(), "offline: bag done in %.1f min wall (%.1fx real "
    "time): %zu scans seen, %zu processed, %zu without odom pose, %zu "
    "nodes added, %zu priors ingested, %d priors attached",
    wall_s / 60.0, bag_ns / 1.0e9 / wall_s, scans_seen_, scans_processed_,
    scans_no_odom_, nodes_added_, priors_ingested_, attached_priors_);

  return nodes_added_ > 0;
}

/*****************************************************************************/
bool OfflineSlamToolbox::finishAndSave()
/*****************************************************************************/
{
  {
    // final solve so the last partial prior_optimize_every_n_nodes window
    // is pulled onto the anchors (the async pipeline gets this from the
    // post-playback settle)
    boost::mutex::scoped_lock lock(smapper_mutex_);
    if (smapper_ && smapper_->getMapper() &&
      smapper_->getMapper()->GetGraph() &&
      !smapper_->getMapper()->GetGraph()->GetVertices().empty())
    {
      RCLCPP_INFO(get_logger(), "offline: final pose-graph optimization...");
      smapper_->getMapper()->CorrectPoses();
    }
  }

  if (offline_output_basename_.empty()) {
    RCLCPP_WARN(get_logger(), "offline: offline_output_basename not set; "
      "skipping serialization and map export.");
    return true;
  }

  RCLCPP_INFO(get_logger(), "offline: serializing pose graph to %s",
    offline_output_basename_.c_str());
  auto req =
    std::make_shared<slam_toolbox::srv::SerializePoseGraph::Request>();
  auto resp =
    std::make_shared<slam_toolbox::srv::SerializePoseGraph::Response>();
  req->filename = offline_output_basename_;
  serializePoseGraphCallback(nullptr, req, resp);
  if (resp->result !=
    slam_toolbox::srv::SerializePoseGraph::Response::RESULT_SUCCESS)
  {
    RCLCPP_ERROR(get_logger(), "offline: pose graph serialization failed.");
    return false;
  }

  return saveMap(offline_output_basename_);
}

/*****************************************************************************/
bool OfflineSlamToolbox::saveMap(const std::string & basename)
/*****************************************************************************/
{
  // in-process equivalent of nav2 map_saver_cli (same trinary pgm + yaml
  // conventions) so no second node or DDS transport is needed
  nav_msgs::msg::OccupancyGrid map;
  {
    boost::mutex::scoped_lock lock(smapper_mutex_);
    karto::OccupancyGrid * occ_grid = smapper_->getOccupancyGrid(resolution_);
    if (!occ_grid) {
      RCLCPP_ERROR(get_logger(), "offline: could not build occupancy grid.");
      return false;
    }
    vis_utils::toNavMap(occ_grid, map);
    delete occ_grid;
  }

  const std::string pgm_path = basename + ".pgm";
  std::ofstream pgm(pgm_path, std::ios::binary);
  if (!pgm) {
    RCLCPP_ERROR(get_logger(), "offline: cannot write %s", pgm_path.c_str());
    return false;
  }
  pgm << "P5\n" << map.info.width << " " << map.info.height << "\n255\n";
  // pgm rows run top-down, grid data runs bottom-up
  for (int y = static_cast<int>(map.info.height) - 1; y >= 0; y--) {
    for (unsigned int x = 0; x < map.info.width; x++) {
      const int8_t value = map.data[map.info.width * y + x];
      unsigned char pixel = 205;  // unknown
      if (value == 0) {
        pixel = 254;  // free
      } else if (value == 100) {
        pixel = 0;  // occupied
      }
      pgm.put(static_cast<char>(pixel));
    }
  }
  pgm.close();
  if (pgm.fail()) {
    RCLCPP_ERROR(get_logger(), "offline: writing %s failed (disk full?)",
      pgm_path.c_str());
    return false;
  }

  const std::string yaml_path = basename + ".yaml";
  std::ofstream yaml(yaml_path);
  if (!yaml) {
    RCLCPP_ERROR(get_logger(), "offline: cannot write %s", yaml_path.c_str());
    return false;
  }
  const std::string image = pgm_path.find_last_of('/') == std::string::npos ?
    pgm_path : pgm_path.substr(pgm_path.find_last_of('/') + 1);
  yaml << "image: " << image << "\n"
       << "mode: trinary\n"
       << "resolution: " << resolution_ << "\n"
       << "origin: [" << map.info.origin.position.x << ", "
       << map.info.origin.position.y << ", 0]\n"
       << "negate: 0\n"
       << "occupied_thresh: 0.65\n"
       << "free_thresh: 0.25\n";
  yaml.close();
  if (yaml.fail()) {
    RCLCPP_ERROR(get_logger(), "offline: writing %s failed (disk full?)",
      yaml_path.c_str());
    return false;
  }

  RCLCPP_INFO(get_logger(), "offline: wrote %s (%u x %u @ %.2f m) and %s",
    pgm_path.c_str(), map.info.width, map.info.height, resolution_,
    yaml_path.c_str());
  return true;
}

}  // namespace slam_toolbox
