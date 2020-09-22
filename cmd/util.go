package cmd

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

// ClosestHeaderName returns a header name containing <name>
func ClosestHeaderName(fname string, name string) (string, error) {
	f, err := os.Open(fname)
	if err != nil {
		return "", err
	}
	defer f.Close()
	reader := csv.NewReader(f)
	header, err := reader.Read()
	if err != nil {
		return "", err
	}

	for i := range header {
		str := strings.TrimSpace(header[i])
		str = strings.Trim(str, "#/\n")

		if strings.Contains(str, name) {
			return header[i], nil
		}
	}
	msg := fmt.Sprintf("No header contains %s\n", name)
	return "", errors.New(msg)
}

// ReadCoeffs return a map with the coefficients
func ReadCoeffs(fname string) (map[string]float64, error) {
	f, err := os.Open(fname)
	coeff := make(map[string]float64)
	if err != nil {
		return coeff, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	for {
		line, err := reader.Read()
		if err == io.EOF {
			return coeff, nil
		}
		num, err := strconv.ParseFloat(line[1], 64)
		if err != nil {
			return coeff, err
		}
		coeff[line[0]] = num
	}
	return coeff, nil
}

// ReadModel reads a model from a JSON file
func ReadModel(fname string) (Model, error) {
	jsonFile, err := os.Open(fname)
	if err != nil {
		return Model{}, nil
	}
	defer jsonFile.Close()
	bytes, err := ioutil.ReadAll(jsonFile)

	if err != nil {
		return Model{}, err
	}

	var model Model
	json.Unmarshal(bytes, &model)
	return model, nil
}

// Score is a conveniene type used to collect information about the quality of a model
type Score struct {
	Name  string
	Value float64
}

// Model is convenience type used to store information about a model
type Model struct {
	Datafile string
	Coeffs   map[string]float64
	Score    Score
}
