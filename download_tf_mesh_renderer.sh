#!/bin/bash
export TEST_SRCDIR=`pwd`

git clone https://github.com/google/tf_mesh_renderer
cd tf_mesh_renderer

# Restore to a specific version of tf_mesh_renderer.
git reset --hard 8f851958c1548ba7b5cbaa074fa5f7b3b9331b0d
cp ../mesh_renderer_tf2_upgrade.patch .
git apply mesh_renderer_tf2_upgrade.patch

# Use gcc to build kernels.
cd mesh_renderer/kernels
TF_COMPILEFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LINKFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
export LD_LIBRARY_PATH=$TF_LIB:$LD_LIBRARY_PATH

g++ -std=c++11 -shared \
  rasterize_triangles_grad.cc rasterize_triangles_op.cc rasterize_triangles_impl.cc rasterize_triangles_impl.h  \
  -o rasterize_triangles_kernel.so -fPIC $TF_COMPILEFLAGS $TF_LINKFLAGS -O2

export PYTHONPATH="$TEST_SRCDIR/tf_mesh_renderer/mesh_renderer:$PYTHONPATH"
cd $TEST_SRCDIR
