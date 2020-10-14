package gafit

import (
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// CostFunctionIdentifier is a pattarn that is search for in the output of the hook
// the floating point number that follows is extracted
const CostFunctionIdentifier = "GOGAFIT_COST:"

// CaptureResult is a type used to represent results captured from
type CaptureResult struct {
	Floats  map[string]float64
	Ints    map[string]int
	Strings map[string]string
}

// NewCaptureResult returns a new initialized instance of CaptureResult
func NewCaptureResult() CaptureResult {
	return CaptureResult{
		Floats:  make(map[string]float64),
		Ints:    make(map[string]int),
		Strings: make(map[string]string),
	}
}

// GetFloat returns captured float values
func (cr CaptureResult) GetFloat(name string) float64 {
	return cr.Floats[name]
}

// GetInt returns captured int values
func (cr CaptureResult) GetInt(name string) int {
	return cr.Ints[name]
}

// GetString return captured string values
func (cr CaptureResult) GetString(name string) string {
	return cr.Strings[name]
}

// CaptureFunction is a type used to capture results from a string
type CaptureFunction func(out string) (CaptureResult, error)

// Hook is a type that runs the script and capture results from the output using the
// Capture function
type Hook struct {
	Script  string
	Capture CaptureFunction
}

// CostFunctionHook is a type used to represent external cost functions
type CostFunctionHook struct {
	Hook Hook
}

// NewCostFunctionHook returns a new instance of a cost function
func NewCostFunctionHook(script string) CostFunctionHook {
	return CostFunctionHook{
		Hook: Hook{
			Script:  script,
			Capture: captureCostFuncValue,
		},
	}
}

// FittedModel is a type that holds the design matrix, the target values
// and the coefficients
type FittedModel struct {
	Rows  int
	Cols  int
	X     []float64
	Y     []float64
	Coeff []float64
	Names []string
}

func model2json(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense, names []string) string {
	r, c := X.Dims()
	model := FittedModel{
		Rows:  r,
		Cols:  c,
		X:     X.RawMatrix().Data,
		Y:     y.RawVector().Data,
		Coeff: coeff.RawVector().Data,
		Names: names,
	}
	b, err := json.Marshal(&model)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// Execute runs
func (cfh CostFunctionHook) Execute(X *mat.Dense, y *mat.VecDense, coeff *mat.VecDense, names []string) float64 {
	strRep := model2json(X, y, coeff, names)

	cmd := exec.Command(cfh.Hook.Script, strRep)
	out, err := cmd.Output()
	if err != nil {
		msg := fmt.Sprintf("Error when running script: %s\n", err)
		panic(msg)
	}
	outStr := string(out)
	res, err := cfh.Hook.Capture(outStr)

	if err != nil {
		panic(err)
	}

	return res.GetFloat("cost")
}

func captureCostFuncValue(out string) (CaptureResult, error) {
	result := NewCaptureResult()
	prog := regexp.MustCompile(CostFunctionIdentifier + " ([[+-]?[0-9]*\\.?[0-9]+)")

	res := prog.FindStringSubmatch(out)
	if len(res) < 2 {
		return result, errors.New("No match found")
	}

	f, err := strconv.ParseFloat(res[1], 64)
	if err != nil {
		return result, err
	}
	result.Floats["cost"] = f
	return result, nil
}
