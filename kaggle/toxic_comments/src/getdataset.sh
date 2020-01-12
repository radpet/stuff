kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p ./data/

for f in data/*
do
  unzip $f -d ./data
done