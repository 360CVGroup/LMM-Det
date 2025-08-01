# LMM-Det: Make Large Multimodal Models Excel in Object Detection

This repository is the official implementation of **LMM-Det**,  a simple yet effective approach that leverages a Large Multimodal Model for vanilla object Detection without relying on specialized detection modules.

**[LMM-Det: Make Large Multimodal Models Excel in Object Detection](https://arxiv.org/abs/2507.18300)** 
</br>
Jincheng Li*, Chunyu Xie*, Ji Ao, Dawei Leng†, Yuhui Yin (*Equal Contribution, ✝Corresponding Author)
</br>

## 中文
[我爱计算机视觉​](https://zhuanlan.zhihu.com/p/1932117466013758166)

[360AI研究院](https://research.360.cn/blog/detail/6876171cb8ed93721b543154)


## 🔥 News
- 🚀 **[2025/08/01]** We have updated the LMM-Det github repository, and now you can test our models!
- 🚀 **[2025/07/24]** We released the paper of [LMM-Det: Make Large Multimodal Models Excel in Object Detection](https://arxiv.org/abs/2507.18300).
- 🚀 **[2025/06/26]** LMM-Det has been accepted by ICCV'25.

## Contents
- [Install](#install)
- [Model Zoo](#modelzoo)
- [Customized Dataset](#customizeddataset)
- [Preparation](#customizeddataset)
- [Evaluation](#evaluation)



## Install

```Shell
# remember to modify Line 7 in deploy.sh 
bash deploy.sh
```

## Model Zoo

[🤗LMM-Det-StageIV](https://huggingface.co/qihoo360/LMM-Det/tree/main/checkpoints)

[🤗OWLv2-ViT](https://huggingface.co/google/owlv2-large-patch14-ensemble)

We also provide the official weight of [OWlv2-ViT](https://huggingface.co/qihoo360/LMM-Det/tree/main/checkpoints/owlv2-large-patch14-ensemble)

## Customized Dataset

We have released our [customized dataset](https://huggingface.co/qihoo360/LMM-Det/tree/main/custom_data) during Stage IV.

For the curation details, please refer to: [[Custom Data](custom_data/custom_data.md)]


## Preparation

Step 1: Download the [COCO](https://cocodataset.org/) dataset. You can put COCO into LMM-Det/data or make a soft link using ln -s.

Step 2: Modify the COCO Path in Lines 4-5 in LMM-Det/scripts/eval/eval_coco_model_w_sft_data.sh


Step 3: Download the model and put it into LMM-Det/checkpoints


## Evaluation


```Shell 
bash evaluate.sh
```

## We Are Hiring
We are seeking academic interns in the Multimodal field. If interested, please send your resume to xiechunyu@360.cn.



## BibTeX
```
@misc{li2025lmmdet,
      title={LMM-Det: Make Large Multimodal Models Excel in Object Detection}, 
      author={Jincheng Li and Chunyu Xie and Ji Ao and Dawei Leng and Yuhui Yin},
      year={2025},
      eprint={2507.18300},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.18300}, 
}
```


## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).


## Related Projects
This work wouldn't be possible without the incredible open-source code of these projects. Huge thanks!
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [OWLv2](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/owlv2.md)
- [Salience-DETR](https://github.com/xiuqhou/Salience-DETR)