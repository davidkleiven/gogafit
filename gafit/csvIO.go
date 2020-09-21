package gafit

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Read dataset from the a file
func Read(fname string, targetName string) (Dataset, error) {
	f, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	return ReadFile(f, targetName)
}

// ReadFile creates a dataset from the passed file
func ReadFile(csvfile *os.File, targetName string) (Dataset, error) {
	r := csv.NewReader(csvfile)
	data := Dataset{}

	isHeader := true
	targetCol := 0
	X := []float64{}
	y := []float64{}
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}

		if isHeader {
			isHeader = false
			parseHeader(record)
			record, targetCol = remove(record, targetName)

			if targetCol == -1 {
				return data, errors.New("Target column not found in the header")
			}
			data.ColNames = record
			data.TargetName = targetName
		} else {
			values, err := parseValues(record)
			if err != nil {
				return data, err
			}
			values, targ := removeIndex(values, targetCol)
			y = append(y, targ)
			for _, v := range values {
				X = append(X, v)
			}
		}
	}

	// Populate the dataset
	nr, nc := len(y), len(X)/len(y)

	if len(X) != nr*nc {
		msg := fmt.Sprintf("Dimension mismatch. Expected %d elements got %d\n", nr*nc, len(X))
		return data, errors.New(msg)
	}
	data.X = mat.NewDense(nr, nc, X)
	data.Y = mat.NewVecDense(nr, y)
	return data, nil
}

func parseHeader(record []string) {
	for i := range record {
		record[i] = strings.Trim(record[i], "#/ \n\t\r\v")
	}
}

func parseValues(record []string) ([]float64, error) {
	res := make([]float64, len(record))
	for i := range record {
		str := strings.TrimSpace(record[i])
		v, err := strconv.ParseFloat(str, 64)
		if err != nil {
			return res, err
		}
		res[i] = v
	}
	return res, nil
}

func remove(record []string, toRemove string) ([]string, int) {
	// Find position
	pos := -1
	for i := range record {
		if record[i] == toRemove {
			pos = i
			break
		}
	}

	if pos == -1 {
		return record, pos
	}

	// Remove item and preserve order
	copy(record[pos:], record[pos+1:])
	record = record[:len(record)-1]
	return record, pos
}

// Remove item at position idx, and return the value of the removed item
// Order is preserved
func removeIndex(x []float64, idx int) ([]float64, float64) {
	value := x[idx]
	copy(x[idx:], x[idx+1:])
	x = x[:len(x)-1]
	return x, value
}
