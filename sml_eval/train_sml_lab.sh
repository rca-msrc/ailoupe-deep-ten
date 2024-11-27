 # ("sml_lab", "single", "test", 1000, transform_val),
# ("sml_lab_test", "single", None, None, transform_val),
# ("sml_expo_eval", "single", None, None, transform_val),
# ("sml_lab", "dual", "train", 1000, transform_train),
# ("sml_lab", "dual", "test", 1000, transform_val),
# ("sml_lab_test", "dual", None, None, transform_val),
# ("sml_expo_eval", "dual", None, None, transform_val)




# python3 train_dual.py --model resnet50 --dataset sml_lab_dual --checkname dual_image_model --batch-size 512 --test-batch-size 512 --lr 0.1 --epochs 30 --train-n-per-class 1000 --pairing-strategy product --include-inverse --resume runs/sml_lab_dual/resnet50/dual_image_model/model_best.pth.tar --plot


# baseline
#python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_check --lr 0.025 --batch-size 64
