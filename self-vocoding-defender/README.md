Before watermark embedding and extraction, download the pre-trained checkpoint from [link](https://github.com/TimbreWatermarking/TimbreWatermarking)
### Watermark embedding

```
python embed_and_save.py --wm index\in\results\wmpool.txt -o path\to\clean\wavs -s path\to\save\watermarked\wavs -mp path\to\model\directory -p config/process.yaml -m config/model.yaml -t config/train.yaml
```


### Watermark extracting

```
python extract.py --wm index\in\results\wmpool.txt -p config/process.yaml -m config/model.yaml -t config/train.yaml -mp path\to\model\directory -tp \path\to\wavs\or\spectrogram\being\decoded
```
