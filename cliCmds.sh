FOLDER="gafit/_testdata"
DATAFILE="${FOLDER}/dataset.csv"

echo "Testing fit"
go run main.go fit -d $DATAFILE -y Var4 -g 5 -o coeff.csv

echo "Test RMSE"
go run main.go rmse -d $DATAFILE -y Var4 -c coeff.csv
rm coeff.csv

echo "Test ttsplit"
go run main.go ttsplit -d $DATAFILE -f 0.2
rm "${FOLDER}/dataset_train.csv"
rm "${FOLDER}/dataset_test.csv"

echo "Testing ELM command"
go run main.go elm -d $DATAFILE -y Var4 -r 20 -s 10
rm "${FOLDER}/dataset_elm.csv"