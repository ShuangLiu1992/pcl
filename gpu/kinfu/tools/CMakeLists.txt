
file(GLOB hdrs "*.h*")

## recorder
set(the_target pcl_kinfu_recorder)
set(srcs kinfu_recorder.cpp)

source_group("Source Files" FILES ${srcs})

PCL_ADD_EXECUTABLE(${the_target} COMPONENT ${SUBSYS_NAME} SOURCES ${srcs} ${hdrs} BUNDLE)
target_link_libraries("${the_target}" pcl_io ${OPENNI_LIBRARIES} pcl_gpu_kinfu)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  target_link_libraries("${the_target}" ${OpenCV_LIBS})
  target_compile_definitions("${the_target}" PRIVATE -DHAVE_OPENCV)
endif()

find_package(realsense2 REQUIRED)
target_include_directories("${the_target}" PRIVATE ${realsense2_INCLUDE_DIR})
target_link_libraries("${the_target}" ${realsense2_LIBRARY})

## live
set(the_target pcl_kinfu_live)
set(srcs kinfu_live.cpp)

source_group("Source Files" FILES ${srcs})

PCL_ADD_EXECUTABLE(${the_target} COMPONENT ${SUBSYS_NAME} SOURCES ${srcs} ${hdrs} BUNDLE)
target_link_libraries("${the_target}" pcl_io ${OPENNI_LIBRARIES} pcl_gpu_kinfu)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  target_link_libraries("${the_target}" ${OpenCV_LIBS})
  target_compile_definitions("${the_target}" PRIVATE -DHAVE_OPENCV)
endif()

find_package(realsense2 REQUIRED)
target_include_directories("${the_target}" PRIVATE ${realsense2_INCLUDE_DIR})
target_link_libraries("${the_target}" ${realsense2_LIBRARY})

## eval
set(the_target pcl_kinfu_eval)
set(srcs kinfu_eval.cpp)

source_group("Source Files" FILES ${srcs})

PCL_ADD_EXECUTABLE(${the_target} COMPONENT ${SUBSYS_NAME} SOURCES ${srcs} ${hdrs} BUNDLE)
target_link_libraries("${the_target}" pcl_io ${OPENNI_LIBRARIES} pcl_gpu_kinfu)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  target_link_libraries("${the_target}" ${OpenCV_LIBS})
  target_compile_definitions("${the_target}" PRIVATE -DHAVE_OPENCV)
endif()

find_package(realsense2 REQUIRED)
target_include_directories("${the_target}" PRIVATE ${realsense2_INCLUDE_DIR})
target_link_libraries("${the_target}" ${realsense2_LIBRARY})