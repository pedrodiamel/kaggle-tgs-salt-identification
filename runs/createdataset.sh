

PATH=~/.kaggle/tgs-salt-identification-challenge/
kaggle competitions download -c tgs-salt-identification-challenge
unzip $PATH/*.zip
mkdir $PATH/test
mv  $PATH/images  $PATH/test/

