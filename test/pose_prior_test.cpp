/*
 * pose_prior_test
 * Copyright (c) 2026, Pixel Robotics GmbH
 *
 * THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
 * COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
 * COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS
 * AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
 */

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "gtest/gtest.h"
#include "karto_sdk/Karto.h"
#include "solvers/ceres_utils.h"

namespace
{

Eigen::Matrix3d SqrtInformationFromCovariance(
  const Eigen::Matrix3d & covariance, bool position_only)
{
  Eigen::Matrix3d sqrt_information = Eigen::Matrix3d::Zero();
  if (position_only) {
    sqrt_information.topLeftCorner<2, 2>() =
      Eigen::Matrix2d(covariance.topLeftCorner<2, 2>().inverse())
      .llt().matrixU();
  } else {
    sqrt_information = Eigen::Matrix3d(covariance.inverse()).llt().matrixU();
  }
  return sqrt_information;
}

}  // namespace

TEST(PosePrior2dErrorTerm, FullPoseResidual)
{
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 0.01;
  const Eigen::Matrix3d sqrt_information =
    SqrtInformationFromCovariance(covariance, false);

  PosePrior2dErrorTerm term(1.0, 2.0, 0.5, sqrt_information);
  const double x = 1.3, y = 1.6, yaw = 0.7;
  double residuals[3];
  ASSERT_TRUE(term(&x, &y, &yaw, residuals));

  // sqrt information of 0.01 * I is 10 * I
  EXPECT_NEAR(residuals[0], 10.0 * (1.3 - 1.0), 1e-9);
  EXPECT_NEAR(residuals[1], 10.0 * (1.6 - 2.0), 1e-9);
  EXPECT_NEAR(residuals[2], 10.0 * (0.7 - 0.5), 1e-9);
}

TEST(PosePrior2dErrorTerm, AngleWrapAround)
{
  const Eigen::Matrix3d sqrt_information = Eigen::Matrix3d::Identity();
  PosePrior2dErrorTerm term(0.0, 0.0, M_PI - 0.1, sqrt_information);
  const double x = 0.0, y = 0.0, yaw = -M_PI + 0.1;
  double residuals[3];
  ASSERT_TRUE(term(&x, &y, &yaw, residuals));
  EXPECT_NEAR(residuals[2], 0.2, 1e-9);
}

TEST(PosePrior2dErrorTerm, PositionOnlyIgnoresYaw)
{
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 0.01;
  covariance(2, 2) = 1e6;  // position-only sentinel
  const Eigen::Matrix3d sqrt_information =
    SqrtInformationFromCovariance(covariance, true);

  PosePrior2dErrorTerm term(1.0, 2.0, 0.0, sqrt_information);
  const double x = 1.3, y = 1.6, yaw = 2.9;
  double residuals[3];
  ASSERT_TRUE(term(&x, &y, &yaw, residuals));

  EXPECT_NEAR(residuals[0], 10.0 * (1.3 - 1.0), 1e-9);
  EXPECT_NEAR(residuals[1], 10.0 * (1.6 - 2.0), 1e-9);
  EXPECT_NEAR(residuals[2], 0.0, 1e-9);
}

TEST(PosePriorGraph, PriorsAnchorTheGraph)
{
  // three nodes chained by exact unit-x odometry edges, two position-only
  // priors offset by (10, 5) from the initial guesses: the optimum keeps the
  // relative geometry and translates the chain onto the priors
  std::vector<Eigen::Vector3d> nodes = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};

  ceres::Problem problem;
  const Eigen::Matrix3d odom_sqrt_info = Eigen::Matrix3d::Identity() * 10.0;
  for (int i = 0; i < 2; i++) {
    problem.AddResidualBlock(
      PoseGraph2dErrorTerm::Create(1.0, 0.0, 0.0, odom_sqrt_info), nullptr,
      &nodes[i](0), &nodes[i](1), &nodes[i](2),
      &nodes[i + 1](0), &nodes[i + 1](1), &nodes[i + 1](2));
  }

  Eigen::Matrix3d prior_covariance = Eigen::Matrix3d::Identity() * 0.01;
  prior_covariance(2, 2) = 1e6;
  const Eigen::Matrix3d prior_sqrt_info =
    SqrtInformationFromCovariance(prior_covariance, true);
  problem.AddResidualBlock(
    PosePrior2dErrorTerm::Create(10.0, 5.0, 0.0, prior_sqrt_info), nullptr,
    &nodes[0](0), &nodes[0](1), &nodes[0](2));
  problem.AddResidualBlock(
    PosePrior2dErrorTerm::Create(12.0, 5.0, 0.0, prior_sqrt_info), nullptr,
    &nodes[2](0), &nodes[2](1), &nodes[2](2));

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  ASSERT_TRUE(summary.IsSolutionUsable());

  EXPECT_NEAR(nodes[0](0), 10.0, 1e-3);
  EXPECT_NEAR(nodes[0](1), 5.0, 1e-3);
  EXPECT_NEAR(nodes[1](0), 11.0, 1e-3);
  EXPECT_NEAR(nodes[1](1), 5.0, 1e-3);
  EXPECT_NEAR(nodes[2](0), 12.0, 1e-3);
  EXPECT_NEAR(nodes[2](1), 5.0, 1e-3);
}

TEST(LocalizedRangeScanPrior, SetShiftClear)
{
  karto::LocalizedRangeScan scan(karto::Name("test_laser"), {1.0, 2.0, 3.0});
  EXPECT_FALSE(scan.HasPosePrior());

  scan.SetPosePrior(
    karto::Pose2(1.0, 2.0, 0.3), karto::Matrix3(), "camera_7",
    karto::Vector2<kt_double>(5.0, 6.0));
  EXPECT_TRUE(scan.HasPosePrior());
  EXPECT_EQ(scan.GetPriorSourceId(), "camera_7");

  // camera_7 re-surveyed 0.5 m east: prior and stored anchor shift with it
  scan.ShiftPosePrior(karto::Vector2<kt_double>(0.5, 0.0));
  EXPECT_NEAR(scan.GetPriorPose().GetX(), 1.5, 1e-9);
  EXPECT_NEAR(scan.GetPriorPose().GetY(), 2.0, 1e-9);
  EXPECT_NEAR(scan.GetPriorPose().GetHeading(), 0.3, 1e-9);
  EXPECT_NEAR(scan.GetPriorAnchorPosition().GetX(), 5.5, 1e-9);
  EXPECT_NEAR(scan.GetPriorAnchorPosition().GetY(), 6.0, 1e-9);

  scan.ClearPosePrior();
  EXPECT_FALSE(scan.HasPosePrior());
}

TEST(LocalizedRangeScanPrior, SerializationRoundTrip)
{
  karto::LocalizedRangeScan scan(karto::Name("test_laser"), {1.0, 2.0, 3.0});
  scan.SetOdometricPose(karto::Pose2(0.1, 0.2, 0.3));
  scan.SetCorrectedPose(karto::Pose2(0.4, 0.5, 0.6));
  karto::Matrix3 covariance;
  covariance.SetToIdentity();
  covariance(0, 0) = 0.01;
  covariance(1, 1) = 0.02;
  covariance(2, 2) = 1e6;
  scan.SetPosePrior(
    karto::Pose2(7.0, 8.0, 0.9), covariance, "camera_42",
    karto::Vector2<kt_double>(3.0, 4.0));

  std::stringstream buffer;
  {
    boost::archive::binary_oarchive oa(buffer);
    oa << scan;
  }

  karto::LocalizedRangeScan restored;
  {
    boost::archive::binary_iarchive ia(buffer);
    ia >> restored;
  }

  ASSERT_TRUE(restored.HasPosePrior());
  EXPECT_NEAR(restored.GetPriorPose().GetX(), 7.0, 1e-9);
  EXPECT_NEAR(restored.GetPriorPose().GetY(), 8.0, 1e-9);
  EXPECT_NEAR(restored.GetPriorPose().GetHeading(), 0.9, 1e-9);
  EXPECT_NEAR(restored.GetPriorCovariance()(0, 0), 0.01, 1e-9);
  EXPECT_NEAR(restored.GetPriorCovariance()(2, 2), 1e6, 1e-3);
  EXPECT_EQ(restored.GetPriorSourceId(), "camera_42");
  EXPECT_NEAR(restored.GetPriorAnchorPosition().GetX(), 3.0, 1e-9);
  EXPECT_NEAR(restored.GetPriorAnchorPosition().GetY(), 4.0, 1e-9);
}

TEST(LocalizedRangeScanPrior, SerializationWithoutPrior)
{
  karto::LocalizedRangeScan scan(karto::Name("test_laser"), {1.0, 2.0});
  scan.SetOdometricPose(karto::Pose2(0.1, 0.2, 0.3));
  scan.SetCorrectedPose(karto::Pose2(0.1, 0.2, 0.3));

  std::stringstream buffer;
  {
    boost::archive::binary_oarchive oa(buffer);
    oa << scan;
  }

  karto::LocalizedRangeScan restored;
  restored.SetPosePrior(karto::Pose2(), karto::Matrix3());
  {
    boost::archive::binary_iarchive ia(buffer);
    ia >> restored;
  }

  EXPECT_FALSE(restored.HasPosePrior());
  EXPECT_NEAR(restored.GetOdometricPose().GetX(), 0.1, 1e-9);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
