# CRM-Siam Training Tutorial

This implements training of CRM-Siam.
### Add CRM-Siam to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/CRM-Siam:$PYTHONPATH
```

## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT10K](http://got-10k.aitestunion.com/)

## Download pretrained backbones
Download pretrained backbones from [here](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) and put them in `pretrained_models` directory

## Training

To train a model, run `train.py` with the desired configs:

```bash
cd experiments/siamban_r50_l234
```

### Multi-processing Distributed Data Parallel Training

Refer to [Pytorch distributed training](https://pytorch.org/docs/stable/distributed.html) for detailed description.

#### Single node, multiple GPUs (We use 3 GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
```

## Testing

```bash
python -u ../../tools/test-multi.py --snapshot CRM-Siam.pth --config config-CRM-Siam.yaml --dataset MDOT
```

## Evaluation

```bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset MDOT        \ # dataset name
	--num 4 		 \ # number thread to eval
	--tracker_prefix 'ch*'   # tracker_name
```

## Hyper-parameter Search

The tuning toolkit will not stop unless you do.

```bash
python ../../tools/tune.py  \
    --dataset UAV20L  \
    --snapshot snapshot/checkpoint_e20.pth  \
    --gpu_id 0
```

