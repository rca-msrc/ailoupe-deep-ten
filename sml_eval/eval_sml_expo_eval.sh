# NOTE NEED SET --resume TO THE CORRECT PATH

# single image model eval for Deep TEN
python train.py --dataset sml_expo_eval --model resnet50 --mode single --eval --checkname single_deepten_resnet50 --batch-size 64  --resume runs/sml_lab_dual/resnet50/dual_image_model/model_best.pth.tar 

# DUAL image model training for Deep TEN using resnet50 as backbone
#python train.py --dataset sml_expo_eval --model resnet50 --mode dual --eval --checkname dual_deepten_resnet50 --batch-size 64  --resume runs/sml_lab_dual/resnet50/dual_image_model/model_best.pth.tar 

# --resume runs/sml_lab_dual/resnet50/dual_image_model/model_best.pth.tar 
