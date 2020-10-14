FOLDER="gafit/_testdata"
DATAFILE="${FOLDER}/dataset.csv"

echo "Testing fit"
go run main.go fit -d $DATAFILE -y Var4 -g 5 -o coeff.json

echo "Testing pred command"
go run main.go pred -d $DATAFILE -m coeff.json
rm "${FOLDER}/dataset_predictions.csv"

echo "Test RMSE"
go run main.go rmse -d $DATAFILE -y Var4 -m coeff.json

echo "Test poly command"
go run main.go poly -d $DATAFILE -y Var4 -o 3 -p Var
rm "${FOLDER}/dataset_poly.csv"

echo "Test plot command"
go run main.go plot -d $DATAFILE -y Var4 -m coeff.json -o plot.png
rm coeff.json
rm plot.png

echo "Test ttsplit"
go run main.go ttsplit -d $DATAFILE -f 0.2
rm "${FOLDER}/dataset_train.csv"
rm "${FOLDER}/dataset_test.csv"

echo "Testing ELM command"
go run main.go elm -d $DATAFILE -y Var4 -r 20 -s 10
rm "${FOLDER}/dataset_elm.csv"

echo "Test template script"
go run main.go hook -t cost -p python -o myhook.py
go run main.go fit -d $DATAFILE -y Var4 -g 5 -o coeff.json -c ./myhook.py
rm myhook.py