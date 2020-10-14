package gafit

import (
	"bytes"
	"text/template"
)

// DemoCostFunc is a type holds parameters for demo scripts
type DemoCostFunc struct {
	OutputIdentifier string
	PythonExec       string
}

// DemoCostFuncPython generates a demo script for python
func DemoCostFuncPython(pyExec string) (string, error) {
	parameters := DemoCostFunc{
		OutputIdentifier: CostFunctionIdentifier,
		PythonExec:       pyExec,
	}

	const script = `#!/usr/bin/env {{.PythonExec}}
import sys
import json

def main(arg):
	args = json.loads(arg)

	# When gogafit calls this script, args will now contain
	# {
	#     "Rows": <number of rows in X>
	#     "Cols": <number of columns in X>
	#     "X": <Design matrix>
	#     "Y": <Target value>
	#     "Coeff": <Fitted coefficients>
	#     "Names": <List with the name of each feature>
	# }
	# Predictions for Y can be obtained via y_pred = X.dot(Coeff)

    # Do your calculations, and store the result in this variable
	cost_value = 0.6

    # Important: The following print statement must be present
    # if gogafit should be able to extract
	print("{{.OutputIdentifier}} {}".format(cost_value))

main(sys.argv[1])
	`
	tmpl, err := template.New("script").Parse(script)
	if err != nil {
		return "", err
	}
	buf := bytes.NewBufferString("")
	err = tmpl.Execute(buf, parameters)

	if err != nil {
		return buf.String(), err
	}
	return buf.String(), nil
}
