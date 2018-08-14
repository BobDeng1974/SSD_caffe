# SSD_caffe
----
1. `src` 中的 `prior_box_layer.cpp` 是prior box 的产生的源码,  主要包含3个函数

   `LayerSetUp`, `Reshape`, `Forward_cpu ` .  接下来一一讲解函数的用法, 

   第一个函数 `LayerSetUp` 的主要功能是 

   1.选择使用 min_size, max_size 和aspect_ratio  方式还是选择 pro_width , pro_height 这种方式,

   2.参数初始化,  clip , flip , variance 

   大家如果第一次看代码会一头雾水, 我也不灌鸡汤什么耐下心来去看... , 有时候看代码 需要积累很多, 不是一蹴而就的事情, 慢慢磨他, 多去试试这些代码, 例如```CHECK_GT(prior_box_param.min_size(i), 0) << "min_size must be positive.";``` 表示  if`prior_box_param.min_size(i)` > 0   , yes: 继续程序,  no: 打印` "min_size must be positive."` 并且停止程序抛出error,  我也是用ide 试过才知道, 多去尝试.  废话不多说 开始代码.

   ```c++
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
   ```
   
   ```c++
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
   
   ```




2. 本文档主要记录学习caffe 框架下 ssd的源码分析过程            

   未完待续..........                                                                                                                                                                                              

   