This is the code for the following paper:

Cai, Lile, Xun Xu, Jun Hao Liew, and Chuan Sheng Foo. 
"Revisiting Superpixels for Active Learning in Semantic Segmentation With Realistic Annotation Costs." 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10988-10997. 2021.

The code is tested with Tensorflow-1.13.2 with Python 3.6.8 using docker image tensorflow/tensorflow:1.13.2-gpu-py3.

1. Prepare Cityscapes dataset.
The Cityscapes dataset should be put in ./deeplab/datasets/cityscapes. You may refer to the deeplab repo (https://github.com/tensorflow/models/tree/master/research/deeplab) for the details.

2. Prepare the xception-65 model pretrained on ImageNet.
The pretrained model should be put in ./deeplab/models. You can download the weights from the deeplab model zoo.

3. Run python ./scripts/extract_superpixels.py to extract superpixels.

4. Run python ./scripts/extract_rectangles.py to extract rectangles.

5. Run python ./scripts/compute_anno_cost.py to compute the annotation cost for each rectangle using polygon-based annotation.

6. Run ./scripts/write_bash_files.py to generate the bash files. Then run:
bash ./bash_files/job_name.sh to run the experiment.

'Sp+Do+Random': region_type = 'sp', v0

'Sp+Do+Uncertainty': region_type = 'sp', v1, is_bal = False

'Sp+Do+ClassBal': region_type = 'sp', v1, is_bal = True

'Rec+Pr+Random': region_type = 'rec', v0

'Rec+Pr+Uncertainty': region_type = 'rec', v1, is_bal = False

Here is the output for 1 run:

![Alt text](./result.png?raw=true)



