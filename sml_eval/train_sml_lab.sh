# single image model training for Deep TEN using resnet50 as backbone
python train.py --dataset sml_lab --model resnet50 --mode single --lr-scheduler cos --epochs 30 --checkname single_deepten_resnet50 --lr 0.025 --batch-size 64 --train-n-per-class 1000 --test-n-per-class 1000 --plot

# DUAL image model training for Deep TEN using resnet50 as backbone
#python train.py --dataset sml_lab --model resnet50 --mode dual --lr-scheduler cos --epochs 30 --checkname dual_deepten_resnet50 --lr 0.025 --batch-size 64 --train-n-per-class 1000 --test-n-per-class 1000 --plot

# use resume to resume training
# --resume runs/sml_lab_dual/resnet50/dual_image_model/model_best.pth.tar 
