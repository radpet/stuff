kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p ./data/

for f in data/*
do
  unzip $f -d ./data
done

wget -P ./data -O hate_speech.csv https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv