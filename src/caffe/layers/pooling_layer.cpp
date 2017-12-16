#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->pad_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this -> num_spatial_axes_; ++i) {
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_VEC)
        << "Padding implemented only for vector, average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.pooling_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.pooling_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    out_channels_ = channels_;
    in_channels_ = num_output_;
  } else {
    out_channels_ = num_output_;
    in_channels_ = channels_;
  }


}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?

template <typename Dtype>
void PoolingLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_pool_im2col)
{
  const Dtype* col_buff = input;
  if (!is_1x1_){
    if (!skip_pool_im2col) {
      pool_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g=0; g < group_; ++g){
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, out_channels_ / group_, out_spatial_dim_,
        kernel_dim_, (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g, (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias)
{
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_, out_spatial_dim_, 1, (Dtype)1., bias,
      bias_multiplier_.cpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void PoolingLayer<Dtype>::backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input)
{
  Dtype* col_buff =col_buffer_.mutable_cpu_data();
  if (is_1x1_)
    col_buff = input;
  for (int g = 0; g < group_; ++g){
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, out_spatial_dim_, out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g, (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    pool_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights)
{
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    pool_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, out_channels_ / group_, kernel_dim_, out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g, (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY
template <typename Dtype>
void PoolingLayer<Dtype>::forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_pool_im2col)
{
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if(!skip_pool_im2col){
      pool_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, out_channels_ / group_, out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g, (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias)
{
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_, out_spatial_dim_, 1,
      (Dtype)1., bias, bias_multiplier_.gpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void PoolingLayer<Dtype>::backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input)
{
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, out_spatial_dim_, out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g, (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    pool_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights)
{
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    pool_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, out_channels_ / group_, kernel_dim_, out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g, (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::backward_gpu_bias(Dtype* bias, const Dtype* input)
{
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input, bias_multiplier_.gpu_data(), 1., bias);
}
#endif // For GPU

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_VEC: {
      for (int i=0;i<bottom.size();++i){
        bottom_data = bottom[i]->cpu_data();
        for (int n = 0; n < this->num_; ++n){
          this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data +n * this->top_dim_);
          if(this->bias_term_) {
	    const Dtype* bias = this->blobs_[1]->cpu_data();
            this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
          }
        }
      }
      break;
    }
    case PoolingParameter_PoolMethod_MAX: {
      // Initialize
      if (use_top_mask) {
        top_mask = top[1]->mutable_cpu_data();
        caffe_set(top_count, Dtype(-1), top_mask);
      } else {
        mask = max_idx_.mutable_cpu_data();
        caffe_set(top_count, -1, mask);
      }
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);
      // The main loop
      for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = min(hstart + kernel_h_, height_);
              int wend = min(wstart + kernel_w_, width_);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              const int pool_index = ph * pooled_width_ + pw;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width_ + w;
                  if (bottom_data[index] > top_data[pool_index]) {
                    top_data[pool_index] = bottom_data[index];
                    if (use_top_mask) {
                      top_mask[pool_index] = static_cast<Dtype>(index);
                    } else {
                      mask[pool_index] = index;
                    }
                  }
                }
              }
            }
          }
          // compute offset
          bottom_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
          if (use_top_mask) {
            top_mask += top[0]->offset(0, 1);
          } else {
            mask += top[0]->offset(0, 1);
          }
        }
      }
      break;
    }
    case PoolingParameter_PoolMethod_AVE:{
      for (int i = 0; i < top_count; ++i) {
        top_data[i] = 0;
      }
      // The main loop
      for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = min(hstart + kernel_h_, height_ + pad_h_);
              int wend = min(wstart + kernel_w_, width_ + pad_w_);
              int pool_size = (hend - hstart) * (wend - wstart);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              hend = min(hend, height_);
              wend = min(wend, width_);
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  top_data[ph * pooled_width_ + pw] +=
                      bottom_data[h * width_ + w];
                }
              }
              top_data[ph * pooled_width_ + pw] /= pool_size;
            }
          }
          // compute offset
          bottom_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
        }
      }
      break;
    }
    case PoolingParameter_PoolMethod_STOCHASTIC: {
      NOT_IMPLEMENTED;
      break;
    }
    default:{
      LOG(FATAL) << "Unknown pooling method.";
    }
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  //Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_VEC: {
      Dtype* weight_diff = this-> blobs_[1]->mutable_cpu_diff();
      for(int i = 0; i< top.size();++i) {
        const Dtype* top_diff = top[i]->cpu_diff();
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
	if (this->bias_term_ && this->param_propagate_down_[1]) {
	  Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
	  for (int n = 0; n < this->num_; ++n){
	    this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
    	  }
  	}
        if (this->param_propagate_down_[0] || propagate_down[i]) {
          for (int n = 0; n < this->num_; ++n){
            if (this->param_propagate_down_[0]) {
              this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_, top_diff + n * this->top_dim_, weight_diff);
            }
            if (propagate_down[i]) {
              this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight, bottom_diff + n * this->bottom_dim_);
            }
          }
        }
      }
      break;
    }
    case PoolingParameter_PoolMethod_MAX: {
      // The main loop
      if (use_top_mask) {
        top_mask = top[1]->cpu_data();
      } else {
        mask = max_idx_.cpu_data();
      }
      for (int n = 0; n < top[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              const int index = ph * pooled_width_ + pw;
              const int bottom_index =
                  use_top_mask ? top_mask[index] : mask[index];
              bottom_diff[bottom_index] += top_diff[index];
            }
          }
          bottom_diff += bottom[0]->offset(0, 1);
          top_diff += top[0]->offset(0, 1);
          if (use_top_mask) {
            top_mask += top[0]->offset(0, 1);
          } else {
            mask += top[0]->offset(0, 1);
          }
        }
      }
      break;
    }
    case PoolingParameter_PoolMethod_AVE: {
      // The main loop
      for (int n = 0; n < top[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              int hstart = ph * stride_h_ - pad_h_;
              int wstart = pw * stride_w_ - pad_w_;
              int hend = min(hstart + kernel_h_, height_ + pad_h_);
              int wend = min(wstart + kernel_w_, width_ + pad_w_);
              int pool_size = (hend - hstart) * (wend - wstart);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              hend = min(hend, height_);
              wend = min(wend, width_);
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  bottom_diff[h * width_ + w] +=
                    top_diff[ph * pooled_width_ + pw] / pool_size;
                }
              }
            }
          }
          // offset
          bottom_diff += bottom[0]->offset(0, 1);
          top_diff += top[0]->offset(0, 1);
        }
      }
      break;
    }
    case PoolingParameter_PoolMethod_STOCHASTIC: {
      NOT_IMPLEMENTED;
      break;
    }
    default: {
      LOG(FATAL) << "Unknown pooling method.";
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
