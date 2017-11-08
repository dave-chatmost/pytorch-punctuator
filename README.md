Punctuation prediction system implemented by PyTorch.

## Usage
### Train
```shell
$ CUDA_VISIBLE_DEVICES=6 python train_blstm.py --train-data=data/train --cv-data=data/valid --vocab=data/vocab --punc-vocab=data/punc_vocab --batch-size=128 --epochs=7 --early-stop --verbose --print-freq=10 --save-folder=exp/blstm --checkpoint --lr=0.005 > log/blstm &
$ CUDA_VISIBLE_DEVICES=1 python train_truncated_bptt.py --train_data exp-data/train-head40W --cv_data exp-data/valid --vocab exp-data/vocab --punc_vocab exp-data/punc_vocab --embedding_size 100 --hidden_size 100 --proj_size 100 --hidden_layers 1 --batch_size=256 --bptt_step=20 --half_lr --early_stop --epochs 7 --save_folder exp/lstm/h40w-test --checkpoint --verbose --print_freq 100 --lr 0.01 > log/lstm-h40w-test &
```

### Inference
For more detail
```shell
$ python punctuator.py -h
```
Example
```bash
CUDA_VISIBLE_DEVICES=0 python punctuator.py --data data/test.txt --vocab data/vocab --punc_vocab data/punc_vocab --model_path exp/lstm/final.pth.tar --output result.txt --cuda 
```
