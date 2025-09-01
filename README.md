# GraphHyena
GraphHyena

Code for : Big Data 2025 conference paper GraphHyena: A Hybrid Hyena-Driven Framework for Temporal Graph link Prediction (review phase)

### Try running on UCI dataset

We have already kept the dataset in the processed_data folder for the timebeing so it can be run directly 

python train_link_prediction.py --dataset_name uci --model_name GraphHyena --num_runs 1 --gpu 0 --patch_size 1 --max_input_sequence_length 32 --channel_embedding_dim 50 --dropout 0.1 --hyena_dim 256 --hyena_depth 1 --learning_rate 0.0001 --test_interval_epochs 1

# Acknowledgments

We are grateful to the authors of 
[TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs), 
[TGN](https://github.com/twitter-research/tgn), 
[CAWN](https://github.com/snap-stanford/CAW), 
[EdgeBank](https://github.com/fpour/DGB), 
[GraphMixer](https://github.com/CongWeilin/GraphMixer), and
[DyGFormer](https://github.com/yule-BUAA/DyGLib.git) for making their project codes publicly available.

