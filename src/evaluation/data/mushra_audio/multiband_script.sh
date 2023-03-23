#! /bin/bash
python scripts/process.py -i /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/x//multiband0.wav -r /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/y_ref//multiband0.wav -c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt -out_dir /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/st_y_hat/ -out_f_name multiband0.wav 
python scripts/process.py -i /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/x//multiband1.wav -r /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/y_ref//multiband1.wav -c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt -out_dir /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/st_y_hat/ -out_f_name multiband1.wav 
python scripts/process.py -i /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/x//multiband2.wav -r /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/y_ref//multiband2.wav -c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt -out_dir /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/st_y_hat/ -out_f_name multiband2.wav 
python scripts/process.py -i /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/x//multiband3.wav -r /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/y_ref//multiband3.wav -c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt -out_dir /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/st_y_hat/ -out_f_name multiband3.wav 
python scripts/process.py -i /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/x//multiband4.wav -r /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/y_ref//multiband4.wav -c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt -out_dir /home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio/st_y_hat/ -out_f_name multiband4.wav 
