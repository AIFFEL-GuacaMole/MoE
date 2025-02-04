graphgps pretrain model load 

```
# You can download our pretrained GPS-deep (151 MB).
wget https://www.dropbox.com/s/aomimvak4gb6et3/pcqm4m-GPS%2BRWSE.deep.zip
unzip pcqm4m-GPS+RWSE.deep.zip -d pretrained/

# Run inference and official OGB Evaluator.
python main.py --cfg configs/GPS/pcqm4m-GPSdeep-inference.yaml 

# Result files for OGB-LSC Leaderboard.
results/pcqm4m-GPSdeep-inference/0/y_pred_pcqm4m-v2_test-challenge.npz
results/pcqm4m-GPSdeep-inference/0/y_pred_pcqm4m-v2_test-dev.npz

```
