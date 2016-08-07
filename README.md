# torch_utils

* extract_vgg19.lua

  Extract vgg19 fc7 using loadcaffe with options for dropout, center and averaging 10 crops.
  
* word_RNN.lua

  A self-contained somewhat-optimized variable length RNN implementation.
 

* limited_mem_RNN.lua

  A drop-in replacement of word_RNN that has O(T^0.5) memory complexity.
 

* cudnn_RNN.lua

  A cudnn wrapper for variable length sentences. It's somewhat a drop-in replacement of word_RNN for sentence encoding tasks with certain limitations.
 

* RNNUtils.lua

  Functions for sequence processing, RNN forward/backward.