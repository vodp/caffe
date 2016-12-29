#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/infogain_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void InfogainLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss, const Dtype* infogain,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, const int num_labels_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int l = 0;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = 0;
      for (l = 0; l < num_labels_; l++) {
        loss[index] -= infogain[label_value*num_labels_ + l] *
                        log(max(prob_data[n * spatial_dim * num_labels_ + l * spatial_dim + s],
                                Dtype(kLOG_THRESHOLD)));
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* infogain_mat = NULL;
  if (bottom.size() < 3) {
    infogain_mat = infogain_.gpu_data();
  } else {
    infogain_mat = bottom[2]->gpu_data();
  }
  int count = 0;
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  InfogainLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data, infogain_mat,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, num_labels_,   counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void InfogainLossBackwardGPU(const int nthreads, const Dtype* prob_data,
          const Dtype* label, Dtype* bottom_diff, const Dtype* infogain, const Dtype* rows_infogain,
          const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, const int num_labels_, Dtype* counts) {
  //const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      //for (int c = 0; c < channels; ++c) {

      for (int l = 0; l < num_labels_; ++l) {
        bottom_diff[n * dim + l * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      for (int l = 0; l < num_labels_; ++l) {
        bottom_diff[n*dim + l*spatial_dim + s] = 
            prob_data[n*dim + l*spatial_dim + s]*rows_infogain[label_value] -
            infogain[label_value*num_labels_ + s];
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
  //  caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();

    const Dtype* infogain_mat = NULL;
    if (bottom.size() < 3) {
      infogain_mat = infogain_.gpu_data();
    } else {
      infogain_mat = bottom[2]->gpu_data();
      sum_rows_of_H(bottom[2]);
    }
    const Dtype* rows_infogain = sum_rows_H_.gpu_data();
    
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    
    // NOLINT_NEXT_LINE(whitespace/operators)
    InfogainLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, bottom_diff, infogain_mat, rows_infogain,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, num_labels_, counts);

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InfogainLossLayer);

}  // namespace caffe
