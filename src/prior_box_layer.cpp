#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/prior_box_layer.hpp"

namespace caffe {

template <typename Dtype>
void PriorBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const PriorBoxParameter &prior_box_param =
      this->layer_param_.prior_box_param();
  if ((prior_box_param.min_size_size() > 0) && (prior_box_param.pro_width_size() > 0)) {
    LOG(FATAL) << "min_size and pro_width could not be provided at the same time.";
  } else if (prior_box_param.min_size_size() == 0 && prior_box_param.pro_width_size() == 0) {
    LOG(FATAL) << "Must provide min_size or pro_width.";
  }
    stride_ = prior_box_param.stride();//default=1
  // use size and aspect_ratio to define prior-boxes
  if (prior_box_param.min_size_size() > 0) { // 使用min_size_size 方式进行设置
    // min_sizes
    min_size_.clear();
    for (int i = 0; i < prior_box_param.min_size_size(); ++i) {
      CHECK_GT(prior_box_param.min_size(i), 0) << "min_size must be positive.";
      min_size_.push_back(prior_box_param.min_size(i));
    }
    // flip the ar
    flip_ = prior_box_param.flip();


    // aspect_ratios   高宽比
    aspect_ratios_.clear();
    aspect_ratios_.push_back(1.); // 加入1 
    for (int i = 0; i < prior_box_param.aspect_ratio_size(); ++i) {
      float ar = prior_box_param.aspect_ratio(i); // 把所有的长宽比 给了 ar
      bool already_exist = false; // 判断是否把所有的 ratios 获取了
      for (int j = 0; j < aspect_ratios_.size(); ++j) {
        if (fabs(ar - aspect_ratios_[j]) < 1e-6) { // 判断 aspect_ratios_ 只有一个值, fab浮点型绝对值
          already_exist = true;
          break;
        }
      }
      if (!already_exist) {
        aspect_ratios_.push_back(ar);
        if (flip_) {  // 翻转改变长宽比, 
          aspect_ratios_.push_back(1. / ar); //  1,2,3,  1/2 ,1/3
        }
      }
    } // 到这里，共有5个ratios，分别为1,2,1/2,3,1/3 
    // max_scale
    num_priors_ = aspect_ratios_.size() * min_size_.size();  // 5*1 = 5
    if (prior_box_param.max_size_size() > 0) {
      CHECK_EQ(prior_box_param.max_size_size(), min_size_.size())
        << "max_sizes and min_sizes must have the same length.";
      max_size_.clear();
      for (int i = 0; i < prior_box_param.max_size_size(); ++i) {
        CHECK_GT(prior_box_param.max_size(i), min_size_[i]) << "max_size must be greater than min_size.";
        max_size_.push_back(prior_box_param.max_size(i)); // 获取max_size 的值, 
      }
      num_priors_ += max_size_.size(); // num_priors_ = 6;这里很重要，不然就只有5个，和论文中的6个就不相符了
    }
  } else if (prior_box_param.pro_width_size() > 0) { //pro_width_size=6 公式使用直接提供pro_width 的方式, pro_width_size 是这个参数的个数
    CHECK_EQ(prior_box_param.pro_width_size(), prior_box_param.pro_height_size())
      << "pro_width and pro_height must have the same length.";
    pro_widths_.clear(); // 清空
    pro_heights_.clear();
    for (int i = 0; i < prior_box_param.pro_width_size(); ++i) { // 保证每一个 pro_width 都在(0,1)
      CHECK_GT(prior_box_param.pro_width(i),0) << "pro_width must be positive.";
      CHECK_LE(prior_box_param.pro_width(i),1) << "pro_width must be less than 1.";
      pro_widths_.push_back(prior_box_param.pro_width(i));
    }
    for (int i = 0; i < prior_box_param.pro_height_size(); ++i) {
      CHECK_GT(prior_box_param.pro_height(i),0) << "pro_height must be positive.";
      CHECK_LE(prior_box_param.pro_height(i),1) << "pro_height must be less than 1.";
      pro_heights_.push_back(prior_box_param.pro_height(i));
    }
    num_priors_ = pro_widths_.size();  
  } else {
    LOG(FATAL) << "Error: min_sizes / pro_widths are not provided.";
  }
// ***************************** 设置variance
  // output prior-boxes need to be clipped?
  clip_ = prior_box_param.clip();
  // get the boxes code variances
  if (prior_box_param.variance_size() > 1) {
    // Must and only provide 4 variance.
    CHECK_EQ(prior_box_param.variance_size(), 4);
    for (int i = 0; i < prior_box_param.variance_size(); ++i) {
      CHECK_GT(prior_box_param.variance(i), 0); // > 
      variance_.push_back(prior_box_param.variance(i));
    }
  } else if (prior_box_param.variance_size() == 1) { // 如果只有 1  个variance
    CHECK_GT(prior_box_param.variance(0), 0);
    variance_.push_back(prior_box_param.variance(0));
  } else {
    // Set default to 0.1.
    variance_.push_back(0.1);
  }
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  // bottom[0] -> the feature map
  const int layer_width = bottom[0]->width(); // bottom[0]  是feature map, num*128*w*h
  const int layer_height = bottom[0]->height();
  vector<int> top_shape(3, 1);

  top_shape[0] = 1;
  // 2 channels. First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  top_shape[1] = 2;
  top_shape[2] = ((layer_width-1)/stride_+1) * ((layer_height-1)/stride_ +1)* num_priors_ * 4;
  // LOG(INFO)<<layer_width<<"~~"<<layer_width/stride_<<"~~~";
  CHECK_GT(top_shape[2], 0);
  top[0]->Reshape(top_shape); //(num, 2, W*H*Np*4)
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  // get the feature map size
  const int layer_width = bottom[0]->width();
  const int layer_height = bottom[0]->height();
  // get the input image size
  const int img_width = bottom[1]->width();
  const int img_height = bottom[1]->height();
  // get the step size = imgSize/featureSize
  const float step_x = static_cast<float>(img_width) / layer_width; // 为了转换为原图, 得到缩放比例
  const float step_y = static_cast<float>(img_height) / layer_height;
  // get the output data
  Dtype *top_data = top[0]->mutable_cpu_data();
  // channel dim
  int dim = ((layer_width-1)/stride_+1) * ((layer_height-1)/stride_ +1)* num_priors_ * 4;
  // LOG(INFO)<<dim<<"!!!!!!!!!!!!!";
  int idx = 0;
  // now we cal. each locations [output locations]
  for (int h = 0; h < layer_height; h=h+stride_) {  // h is (0, layer_height - 1)
    for (int w = 0; w < layer_width; w=w+stride_) {
      // LOG(INFO)<<"##########"<<h<<"$$"<<w<<"@@"<<layer_height<<"@@"<<layer_width;
      float center_x = (w + 0.5) * step_x;
      float center_y = (h + 0.5) * step_y;// 这里和Faster RCNN 一样，就是把feature map上的点映射回原图,这里加上0.5也是为了四舍五入，和faster rcnn python代码类似
      float box_width, box_height;
      // use min_size_ to define prior-boxes
      if (min_size_.size() > 0) {
        // 使用所有的scale提出boxes
        for (int i = 0; i < min_size_.size(); ++i) {
          // 1
          // 这里的min_size从fc7_mbox_priorbox的60到最后的276，就是s_k从0.2到0.92的过程
           // first prior: aspect_ratio = 1, size = min_size
          box_width = box_height = min_size_[i];
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width; // 除以img_width?
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          //xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          //ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
          // sqrt(min*max)
          if (max_size_.size() > 0) {
            box_width = box_height = sqrt(min_size_[i] * max_size_[i]);
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }
          // aspect_ratios
          for (int r = 0; r < aspect_ratios_.size(); ++r) { // 其他比例计算
            float ar = aspect_ratios_[r]; // 
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size_[i] * sqrt(ar);
            box_height = min_size_[i] / sqrt(ar);
            top_data[idx++] = (center_x - box_width / 2.) / img_width;
            top_data[idx++] = (center_y - box_height / 2.) / img_height;
            top_data[idx++] = (center_x + box_width / 2.) / img_width;
            top_data[idx++] = (center_y + box_height / 2.) / img_height;
          }
        }
      } else if (pro_widths_.size() > 0) {
          CHECK_EQ(pro_widths_.size(),pro_heights_.size());
          for (int i = 0; i < pro_widths_.size(); ++i) {  // 遍历所有的pro 比例
            box_width = pro_widths_[i];
            box_height = pro_heights_[i];
            top_data[idx++] = center_x / img_width  - box_width / 2.;
            top_data[idx++] = center_y / img_height - box_height / 2.;
            top_data[idx++] = center_x / img_width  + box_width / 2.;
            top_data[idx++] = center_y / img_height + box_height / 2.;
          }
      } else {
        LOG(FATAL) << "Error: min_sizes / pro_widths are not provided.";
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip_) {
    for (int d = 0; d < dim; ++d) {  // w*h*Np 总个数
      top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
    }
  }

  // output variances
  //解答： https://github.com/weiliu89/caffe/issues/75
  // 除以variance是对预测box和真实box的误差进行放大，从而增加loss，增大梯度，加快收敛。
  // 另外，top_data += top[0]->offset(0, 1);已经使指针指向新的地址，所以variance不会覆盖前面的结果。
  // offse一般都是4个参数的offset(n,c,w,h),设置相应的参数就可以指到下一张图（以四位张量为例）
  top_data += top[0]->offset(0, 1); // 这里我猜是指向了下一个chanel
  if (variance_.size() == 1) {
    caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
  } else {
    int count = 0;
    for (int h = 0; h < layer_height; h=h+stride_) {
      for (int w = 0; w < layer_width; w=w+stride_) {
        for (int i = 0; i < num_priors_; ++i) {
          for (int j = 0; j < 4; ++j) {
            top_data[count] = variance_[j];
            ++count;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(PriorBoxLayer);
REGISTER_LAYER_CLASS(PriorBox);

} // namespace caffe
