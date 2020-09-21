package cmd

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
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
