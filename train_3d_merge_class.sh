python train_3d_merge_class.py --outdir=./log --data=../../data/ShapeNet/render --gpus=1 --batch=4 --gamma=40 --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --config=pretrained_model/v1-inference.yaml --sd_ckpt=pretrained_model/sd-v1-4.ckpt --resume_pretrain=pretrained_model/shapenet_car.pt