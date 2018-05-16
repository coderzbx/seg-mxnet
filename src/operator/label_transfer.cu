#include "./label_transfer-inl.h"
#include <thrust/sort.h>
#include "./mxnet_op.h"
#include "../common/cuda_utils.h"

#define LABELTRANSFER_CUDA_CHECK(x) \
  /* Code block avoids redefinition of cudaError_t err */ \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    CHECK_EQ(err, cudaSuccess) << "Name: " << #x << " ErrStr:" << cudaGetErrorString(err); \
  } while (0)

namespace mshadow {
namespace cuda {
  template<typename DType>
    __global__ void TransferKernel(DType *dst, 
                                    const DType *src, 
                                    const Tensor<gpu, 1 ,DType> ids, 
                                    const DType ignore_label,
                                    const int sample_out_size, 
                                    const int num) {
      
        CUDA_KERNEL_LOOP(i,static_cast<int>(num)) {
            int n = i%sample_out_size;
            int y = i/sample_out_size;

            const index_t src_id = static_cast<int>(src[y*sample_out_size + n]);
            
            index_t dst_id;
            if (src_id >= static_cast<int>(ids.shape_.Size())) {
              dst_id = static_cast<int>(ignore_label);
            }else{
              dst_id = static_cast<int>(ids[src_id]);
            }
            dst[y*sample_out_size + n] = static_cast<DType>(dst_id);
        }
    }
   
}  // namespace cuda
}  // namespace mshadow

namespace mshadow {
  template<typename DType>
    inline void Transfer(Tensor<gpu, 2, DType> dst, const Tensor<gpu, 2, DType> &src, const Tensor<gpu, 1 ,DType> ids, const DType ignore_label) {

     CHECK_EQ(dst.CheckContiguous(), true);
     cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
     const int num_thread = cuda::kMaxThreadsPerBlock;
     int num = dst.size(0) * dst.size(1);
     dim3 dimBlock(num_thread);
     dim3 dimGrid((num - 1) / num_thread + 1);
     DType *out_ptr = dst.dptr_;
     using namespace mxnet::op::mxnet_op; 
     cuda::CheckLaunchParam(dimGrid, dimBlock, "LabelTransfer Forward");
     
     cuda::TransferKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(out_ptr, src.dptr_, ids, ignore_label, dst.size(1), num);
     LABELTRANSFER_CUDA_CHECK(cudaPeekAtLastError());
    } 
} //namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(LabelTransferParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LabelTransferOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

