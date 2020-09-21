DATAFILE="gafit/_testdata/dataset.csv"

echo "Testing fit"
go run main.go fit -d $DATAFILE -y Var4 -g 5 -o coeff.csv

echo "Test RMSE"
go run main.go rmse -d $DATAFILE -y Var4 -c coeff.csv
rm coeff.csv