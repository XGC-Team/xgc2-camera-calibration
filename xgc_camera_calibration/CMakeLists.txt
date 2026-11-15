cmake_minimum_required(VERSION 3.10)
project(xgc_camera_calibration VERSION 0.3.0)

find_package(catkin REQUIRED)
catkin_python_setup()

# Consume the existing localization wire schema; generate bindings at build time.
find_package(Protobuf REQUIRED)
find_package(xgc2_protobuf CONFIG REQUIRED)
find_package(xgc2_vrpn_zenoh CONFIG REQUIRED)
set(pose_python_dir "${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}")
set(pose_binding "${pose_python_dir}/xgc2/vrpn_zenoh/v1/tracker_pb2.py")
add_custom_command(OUTPUT "${pose_binding}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${pose_python_dir}"
  COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
    "--proto_path=${XGC2_VRPN_ZENOH_PROTO_ROOT}" "--proto_path=${XGC2_PROTOBUF_PROTO_ROOT}"
    "--python_out=${pose_python_dir}"
    "${XGC2_VRPN_ZENOH_PROTO_ROOT}/xgc2/vrpn_zenoh/v1/tracker.proto"
    "${XGC2_PROTOBUF_PROTO_ROOT}/xgc/v1/message.proto"
    "${XGC2_PROTOBUF_PROTO_ROOT}/xgc/semantic/common/v1/types.proto"
  DEPENDS "${XGC2_VRPN_ZENOH_PROTO_ROOT}/xgc2/vrpn_zenoh/v1/tracker.proto"
    "${XGC2_PROTOBUF_PROTO_ROOT}/xgc/v1/message.proto"
    "${XGC2_PROTOBUF_PROTO_ROOT}/xgc/semantic/common/v1/types.proto"
  VERBATIM)
add_custom_target(camera_localization_proto ALL DEPENDS "${pose_binding}")
install(DIRECTORY "${pose_python_dir}/xgc" "${pose_python_dir}/xgc2"
  DESTINATION "${CATKIN_GLOBAL_PYTHON_DESTINATION}" FILES_MATCHING PATTERN "*_pb2.py")
catkin_package()

catkin_install_python(PROGRAMS
  scripts/extrinsic_calibrator_web.py
  scripts/intrinsic_calibrator_web.py
  scripts/extrinsic_tf_publisher.py
  scripts/gazebo_camera_from_extrinsic.py
  scripts/resolve_extrinsic.py
  scripts/render_asset.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(PROGRAMS scripts/resolve_intrinsic.sh
  scripts/validate-scene-replay-asset.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY ../process-definitions
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.json"
)
install(DIRECTORY web
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_asset_document.py)
  catkin_add_nosetests(test/test_solver.py)
  catkin_add_nosetests(test/test_planar_ransac.py)
  catkin_add_nosetests(test/test_intrinsic_solver.py)
  catkin_add_nosetests(test/test_intrinsic_selection.py)
  catkin_add_nosetests(test/test_intrinsic_aprilgrid_truth.py)
  catkin_add_nosetests(test/test_camera_control.py)
  catkin_add_nosetests(test/test_camera_initial_pose.py)
  catkin_add_nosetests(test/test_intrinsic_service.py)
  catkin_add_nosetests(test/test_media_snapshot.py)
  catkin_add_nosetests(test/test_web_service.py DEPENDENCIES camera_localization_proto)
  catkin_add_nosetests(test/test_zenoh_pose.py DEPENDENCIES camera_localization_proto)
  catkin_add_nosetests(test/test_native_web_transport.py)
  catkin_add_nosetests(test/test_native_process_endpoint.py)
  catkin_add_nosetests(test/test_process_contract.py)
  catkin_add_nosetests(test/test_extrinsic_file_watcher.py)
  catkin_add_nosetests(test/test_extrinsic_tf_publisher.py)
  catkin_add_nosetests(test/test_record_facts.py)
  catkin_add_nosetests(test/test_extrinsic_coordinates.py)
  catkin_add_nosetests(test/test_extrinsic_application.py)
  catkin_add_nosetests(test/test_extrinsic_selection.py)
  catkin_add_nosetests(test/test_extrinsic_resolver.py)
endif()
