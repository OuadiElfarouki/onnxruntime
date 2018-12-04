include (ExternalProject)

set(MKLDNN_URL https://github.com/intel/mkl-dnn.git)
# If MKLDNN_TAG is updated, check if platform.cmake.patch needs to be updated.
set(MKLDNN_TAG v0.17.1)
set(MKLDNN_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/src/mkl-dnn/src)
set(MKLDNN_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/install)
set(MKLDNN_LIB_DIR ${MKLDNN_INSTALL}/lib)
set(MKLDNN_INCLUDE_DIR ${MKLDNN_INSTALL}/include)

set(MKLDNN_PATCH_COMMAND1 git apply ${CMAKE_SOURCE_DIR}/patches/mkldnn/platform.cmake.patch)
# discard prior changes due to patching in mkldnn source to unblock incremental builds.
set(MKLDNN_PATCH_DISCARD_COMMAND cd ${MKLDNN_SOURCE} && git checkout -- .)

if(WIN32)
  set(MKLDNN_SHARED_LIB mkldnn.dll)
  set(MKLDNN_IMPORT_LIB mkldnn.lib)
  if(onnxruntime_USE_MKLML)
    set(DOWNLOAD_MKLML ${MKLDNN_SOURCE}/scripts/prepare_mkl.bat)
    set(MKLML_SHARED_LIB mklml.dll)
    set(IOMP5MD_SHARED_LIB libiomp5md.dll)
  endif()
else()
  if (APPLE)
    set(MKLDNN_SHARED_LIB libmkldnn.0.dylib)
  else()
    set(MKLDNN_SHARED_LIB libmkldnn.so.0)
  endif()
  if(onnxruntime_USE_MKLML)
    set(DOWNLOAD_MKLML ${MKLDNN_SOURCE}/scripts/prepare_mkl.sh)
    set(MKLML_SHARED_LIB libmklml_intel.so)
    set(IOMP5MD_SHARED_LIB libiomp5.so)
  endif()
endif()

if(NOT onnxruntime_USE_MKLDNN OR EXISTS ${MKLDNN_SOURCE}/external)
  set(DOWNLOAD_MKLML "")
endif()

ExternalProject_Add(project_mkldnn
    PREFIX mkl-dnn
    GIT_REPOSITORY ${MKLDNN_URL}
    GIT_TAG ${MKLDNN_TAG}
    PATCH_COMMAND ${DOWNLOAD_MKLML} COMMAND ${MKLDNN_PATCH_DISCARD_COMMAND} COMMAND ${MKLDNN_PATCH_COMMAND1}
    SOURCE_DIR ${MKLDNN_SOURCE}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL}
)

if(WIN32)
  add_library(mkldnn STATIC IMPORTED)
  set_property(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_IMPORT_LIB})
else()
  add_library(mkldnn SHARED IMPORTED)
  set_property(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_SHARED_LIB})
endif()
add_dependencies(mkldnn project_mkldnn)
include_directories(${MKLDNN_INCLUDE_DIR})
