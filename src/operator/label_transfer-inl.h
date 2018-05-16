#ifndef MXNET_OPERATOR_LABEL_TRANSFER_INL_H_
#define MXNET_OPERATOR_LABEL_TRANSFER_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "mshadow/tensor.h"


namespace mxnet {
namespace op {

namespace label_transfer_enum {
enum LabelTransferOpInputs {kData};
enum LabelTransferOpOutputs {kOut};
enum LabelTransferNormType {kNull, kBatch, kValid};
enum LabelTransferOpResource {kTempSpace};
}  // namespace label_transfer_enum

struct LabelTransferParam : public dmlc::Parameter<LabelTransferParam> {
    nnvm::Tuple<float> label_ids;
    float ignore_label;
  // float min_kept; // It is a little difficult to implement, and it is not very necessary
  DMLC_DECLARE_PARAMETER(LabelTransferParam) {
    DMLC_DECLARE_FIELD(ignore_label)
    .set_default(255.0f)
    .describe("ignore label id");
    DMLC_DECLARE_FIELD(label_ids)
    .set_default({})
    .describe("label id, ex.{0,1,1,2}");
  };
};

template<typename xpu, typename DType>
class LabelTransferOp : public Operator {
 public:
  explicit LabelTransferOp(LabelTransferParam param) : param_(param), ids_(param.label_ids.begin(), param.label_ids.end()), ignore_label_(param.ignore_label) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U) << "LabelTransfer Input: [label]";
    CHECK_EQ(out_data.size(), 1U) << "LabelTransfer Output: [output]";
    // CHECK_EQ(ids_.size()%2, 0) << "ids count divide by 2";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[label_transfer_enum::kData].size(0);
    int k = in_data[label_transfer_enum::kData].Size()/n;
    Shape<2> s2 = Shape2(n, k);
    Tensor<xpu, 2, DType> data =
            in_data[label_transfer_enum::kData].get_with_shape<xpu, 2, DType>(s2, s);
    Tensor<xpu, 2, DType> out =
            out_data[label_transfer_enum::kOut].get_with_shape<xpu, 2, DType>(s2, s);

    Tensor<xpu, 1 ,DType> ids_workspace =
        ctx.requested[label_transfer_enum::kTempSpace].get_space_typed<xpu, 1, DType>(mshadow::Shape1(ids_.size()), s);
    Tensor<cpu, 1, DType> temp = ctx.requested[label_transfer_enum::kTempSpace].get_host_space_typed<1, DType>(ids_workspace.shape_);
    for (int i=0;i < static_cast<int>(ids_.size()); i++) {
        temp[i] = static_cast<DType>(ids_[i]);
    }

    Copy(ids_workspace, temp, ids_workspace.stream_);
    Transfer(out, data, ids_workspace, static_cast<DType>(ignore_label_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
  }

 private:
  LabelTransferParam param_;
  std::vector<float> ids_;
  float ignore_label_;
};  // class LabelTransferOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(LabelTransferParam param, int dtype);

#if DMLC_USE_CXX11
class LabelTransferProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U) << "Input:[label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;

    // label.shape == data.shape: use probability as label
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LabelTransferProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "LabelTransfer";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  LabelTransferParam param_;
};  // class LabelTransferProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_LABEL_TRANSFER_INL_H_
