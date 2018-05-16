#include "./ohemsoftmax_output-inl.h"
#include <thrust/sort.h>
#include "./mxnet_op.h"
#include "../common/cuda_utils.h"

#define OHEMSOFTMAX_CUDA_CHECK(x) \
  /* Code block avoids redefinition of cudaError_t err */ \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    CHECK_EQ(err, cudaSuccess) << "Name: " << #x << " ErrStr:" << cudaGetErrorString(err); \
  } while (0)

namespace mshadow {
namespace cuda {
template<int n_bits, typename DType>
__global__ void OhemSoftmax3DGradKernel_v1(Tensor<gpu, 3, DType> dst,
                                    const Tensor<gpu, 3, DType> src,
                                    const Tensor<gpu, 2, DType> label,
                                    DType ignore_label,
                                    DType threshold,
                                    DType margin) {
  const index_t xmax = dst.size(1);
  const index_t nmax = dst.size(2);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;
  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    // Is ignore label
    int k = static_cast<int>(label[y][n_index]);
    if (k == static_cast<int>(ignore_label)) {
      for (index_t i = 0; i < xmax; ++i) {
        dst[y][i][n_index] = 0.0f;
      }
      continue;
    }
    // Ohem judge
    const DType predict_score = src[y][k][n_index];
    if (threshold > 0.0f && predict_score > threshold) {
      for (index_t i = 0; i < xmax; ++i) {
        dst[y][i][n_index] = 0.0f;
      }
      continue;
    }
    // Margin Judge
    if (margin > 0.0f && xmax > 2) {
      DType largest = src[y][0][n_index];
      DType second_largest = src[y][1][n_index];
      if (largest < second_largest) {
        largest = src[y][1][n_index];
        second_largest = src[y][0][n_index];
      }
      for (int i = 2; i < xmax; ++i) {
        DType tmp_score = src[y][i][n_index];
        if (tmp_score > largest) {
          second_largest = largest;
          largest = tmp_score;
        } else if (tmp_score > second_largest) {
          second_largest = tmp_score;
        }
      }
      if (predict_score - second_largest >= margin) {
        // Set the gradients to zero
        for (index_t i = 0; i < xmax; ++i) {
          dst[y][i][n_index] = 0.0f;
        }
        continue;
      }
    }
    // Softmax gradient
    for (index_t i = 0; i < xmax; ++i) {
      if (i == k) {
        dst[y][i][n_index] = src[y][i][n_index] - 1.0f;
      } else {
        dst[y][i][n_index] = src[y][i][n_index];
      }
    }
  }
}

template<typename DType>
__global__ void OhemSoftmax3DGradKernel_v2(DType *dst,
                                           const DType *src,
                                           const DType *label,
                                           const DType ignore_label,
                                           const float threshold,
                                           const float margin,
                                           const int class_num,
                                           const int sample_out_size,
                                           const int num) {
  int page_size = sample_out_size * class_num;
  CUDA_KERNEL_LOOP(i, num) {
    int n = i%sample_out_size;
    int y = i/sample_out_size;
    // Get the label of i-th pixel: B(num/sample_out_size) x K(sample_out_size)
    // y * sample_out_size + n == i?
    const index_t k = static_cast<int>(label[y*sample_out_size + n]);
    // Ignore judge
    if (k == static_cast<int>(ignore_label)) {
      for (index_t x = 0; x < class_num; ++x) {
        dst[y*page_size + x*sample_out_size + n] = DType(0.0f);
      }
      continue;
    }
    // Ohem judge
    const float predict_score = static_cast<float>(src[y*page_size + static_cast<int>(k)*sample_out_size + n]);
    if (threshold > 0.0f && predict_score > threshold) {
      for (index_t x = 0; x < class_num; ++x) {
        dst[y*page_size + x*sample_out_size + n] = DType(0.0f);
      }
      continue;
    }
    // Margin Judge
    if (margin > 0.0f && class_num > 2) {
      float largest = src[y*page_size + 0*sample_out_size + n];
      float second_largest = src[y*page_size + 1*sample_out_size + n];
      if (largest < second_largest) {
        largest = src[y*page_size + 1*sample_out_size + n];
        second_largest = src[y*page_size + 0*sample_out_size + n];
      }
      for (int j = 2; j < class_num; ++j) {
        const float tmp_score = src[y*page_size + j*sample_out_size + n];
        if (tmp_score > largest) {
          second_largest = largest;
          largest = tmp_score;
        } else if (tmp_score > second_largest) {
          second_largest = tmp_score;
        }
      }
      if (predict_score - second_largest >= margin) {
        // Set the gradients to zero
        for (index_t x = 0; x < class_num; ++x) {
          dst[y*page_size + x*sample_out_size + n] = DType(0.0f);
        }
        continue;
      }
    }
    // Softmax gradient
    for (index_t x = 0; x < class_num; ++x) {
      if (x == k) {
        dst[y*page_size + x*sample_out_size + n] = src[y*page_size + x*sample_out_size + n] - 1.0f;
      } else {
        dst[y*page_size + x*sample_out_size + n] = src[y*page_size + x*sample_out_size + n];
      } 
    }
  }
}
}  // namespace cuda

#define USE_OHEM_GPU_V1 false

template<typename DType>
inline void OhemSoftmaxGrad(const Tensor<gpu, 3, DType> &dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const DType &ignore_label,
                        const DType &threshold,
                        const DType &margin) {
  CHECK_EQ(dst.CheckContiguous(), true);
  CHECK_EQ(src.CheckContiguous(), true);
  CHECK_EQ(label.CheckContiguous(), true);
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
#if USE_OHEM_GPU_V1
  dim3 dimBlock(cuda::kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "OhemSoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "OhemSoftmaxGrad: label shape mismatch";
  CHECK_EQ(dst.size(2), label.size(1)) << "OhemSoftmaxGrad: label shape mismatch";
  cuda::CheckLaunchParam(dimGrid, dimBlock, "OhemSoftmaxGrad");
  cuda::OhemSoftmax3DGradKernel_v1<cuda::kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(
    dst, src, label, ignore_label, threshold, margin);
  OHEMSOFTMAX_CUDA_CHECK(OhemSoftmax3DGradKernel_v1);
#else
  int num_kernels = dst.size(0) * dst.size(2);
  DType *out_ptr = dst.dptr_;
  using namespace mxnet::op::mxnet_op;
  cuda::OhemSoftmax3DGradKernel_v2<DType><<<cuda_get_num_blocks(num_kernels),
      mshadow::cuda::kBaseThreadNum, 0, stream>>>(out_ptr,
                                                  src.dptr_,
                                                  label.dptr_,
                                                  ignore_label,
                                                  threshold,
                                                  margin,
                                                  dst.size(1),
                                                  dst.size(2),
                                                  num_kernels);
  OHEMSOFTMAX_CUDA_CHECK(OhemSoftmax3DGradKernel_v2);
#endif
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(OhemSoftmaxOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new OhemSoftmaxOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

