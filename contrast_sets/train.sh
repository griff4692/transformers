python ../examples/seq2seq/run_summarization.py \
    --model_name_or_path facebok/bart-large-cnn \
    --do_train \
    --train_file /home/ga2530/contrast-sum/data/train_contrast_sets_small.csv \
    --validation_file /home/ga2530/contrast-sum/data/validation_contrast_sets_small.csv \
    --output_dir /home/ga2530/weights/contrast-sum/ \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --fp16
