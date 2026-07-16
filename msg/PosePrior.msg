# An absolute pose prior for the next pose-graph node, produced by an
# external localization source with surveyed anchors (e.g. ceiling cameras
# or GPS). The pose is the robot base pose in the map frame; the covariance
# marginals (indices 0, 1, 6, 7, 35) weight the prior in the optimization.
# A yaw variance >= 1e5 marks the prior as position-only.
geometry_msgs/PoseWithCovarianceStamped pose

# Identifier of the anchor that produced this prior (e.g. "camera_7").
string source_id

# Optional: world position of that anchor when this prior was computed;
# stored with the node so priors can be shifted if the anchor is later
# re-surveyed (UpdatePriorAnchors service / the offline node's bag
# pre-scan). Leave at (0, 0) when the producer has no survey info -
# unanchored priors are excluded from re-anchoring in both directions.
geometry_msgs/Point anchor_position
