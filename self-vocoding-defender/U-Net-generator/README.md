# Training

```
python train.py --dataroot \dataset\for\pair\training --name model_name --model pix2pix --batch_size 4
```
For dataset creation, please create a directory containing "GT" and "self_vocoded" sub directory, "GT" is for original watermarked wavs and "self_vocoded" is for the corresponding watermark distorted wavs by self-vocoding. The same filename for paired wavs is needed.


# Testing

```
python test.py --dataroot path\to\vocoder\distorted\watermarked\spectrograms --GT_path path\to\watermarked\spectrograms --name model_name --model pix2pix --results_dir path\to\save\output\spectrograms --eval
```
