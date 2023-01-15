Using CPP Whisper: https://github.com/ggerganov/whisper.cpp/issues/89
ffmpeg -i four-score.m4a -ac 2 -ar 16000 -f wav four-score.wav
#time ./main -f ../mlops-template/utils/four-score.wav
time ./main -m models/ggml-medium.en.bin -f ../mlops-template/utils/four-score.wav
##fastest test
time ./main -m models/ggml-medium.en.bin -f ../mlops-template/utils/four-score.wav -t 8 -p 2

