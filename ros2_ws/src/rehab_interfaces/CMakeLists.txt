cmake_minimum_required(VERSION 3.8)
project(rehab_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AdmittanceParameters.msg"
  "msg/EndEffectorState.msg"
  "msg/PolicyAction.msg"
  "msg/RehabTaskState.msg"
  "msg/RealtimeMetrics.msg"
  "msg/SafetyState.msg"
  "msg/WrenchState.msg"
  "srv/EnableRl.srv"
  "srv/LoadPolicy.srv"
  "srv/PauseTask.srv"
  "srv/ResetTask.srv"
  "srv/SetPatientProfile.srv"
  "srv/StartTask.srv"
  "srv/StopTask.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
