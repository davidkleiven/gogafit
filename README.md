# gogafit
Fitting and feature selection based on genetic algorithm.

# CLI Commands
```
gogafit is a tool for training and selecting features for linear models.
Feature selection is done by minimizing diferent cost functions (AICC, BIC, EBIC).
Furthermore, there are a variety of additional tools for common tasks associated
with model selection. Data is represented by simple comma separated files, where
the first line is a header that gives a name to each column.

Usage:
  gogafit [command]

Available Commands:
  elm         Create an extreme learning machine network
  fit         Fit data
  help        Help about any command
  hook        Generate templates scripts for hooks
  plot        Plot the fit in a scatter plot
  poly        Add polynomial versions of a subset of the columns
  pred        Command for predicting from a GA model
  rmse        Calculate RMSE for a model
  ttsplit     Split a dataset in a train and test set

Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
  -h, --help            help for gogafit
  -t, --toggle          Help message for toggle

Use "gogafit [command] --help" for more information about a command.
```
## Fit command
```
Fit linear model from a datafile. Features are selected via a genetic algorithm
by minimizing the given cost function. The data should be organized in a csv file where the
first line is a header which assigns a name to each feature.

Example file
feat1, feat2, feat3
0.1, 0.5, -0.2
0.5, 0.1, 1.0

the program fits a coefficient vector c such that Xc = y. The y vector is called the target vector.
It is extracted from the target parameter passed (e.g. y is the column in the file whose name is
<target>). The remaining columns are taken as the X matrix.

Minimal example:

gogafit fit -d myfile.csv -t feat3

will take the columns corresponding to feat1 and feat2 as the X matrix, and use the last column
(here named feat3) as the y vector.

Usage:
  gogafit fit [flags]

Flags:
  -c, --cost string     Cost function (aic|aicc|bic|ebic) (default "aicc")
  -s, --csplits uint    Number of splits used for cross over operations (default 2)
  -d, --data string     Datafile. Should be stored in CSV format
  -f, --fdratio float   Maximum ratio between number of selected features and number of data points (default 0.8)
  -h, --help            help for fit
  -i, --iprob float     Probability of activating a feature in the initial pool of genomes (default 0.5)
  -r, --lograte uint    Number generation between each log and backup of best solution (default 100)
  -m, --mutrate float   Mutation rate in genetic algorithm (default 0.5)
  -g, --numgen uint     Number of generations to run (default 100)
  -o, --out string      File where the result of the best model is placed (default "model.json")
  -p, --popsize uint    Population size (default 30)
  -y, --target string   Name of the column used as target in the fit (default "lastCol")
  -t, --type string     Fit-type: regression (reg) or classify (cls) (default "reg")

Global Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
```
## TTsplit command
```
Splits the passed csv file into a training set and a validation set.
If the data file is called mydata.csv, the program will create two file

mydata_train.csv for the training data
mydate_test.csv for the test/validation data

Usage:
  gogafit ttsplit [flags]

Flags:
  -d, --data string      Dataset with data
  -f, --fraction float   Fraction of the data placed in the test set (default 0.2)
  -h, --help             help for ttsplit

Global Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
```
## RMSE command
```
Calcualte the root mean square error for a given model.

The prediction is given by p = X.dot(c) where X is the design matrix and y is
c is the coefficient vector. RMSE is given by sqrt(mean((p - y)**2)).

The X and y vector is extracted from a csv file with the format (mydata.csv in the example below)

feat1, feat2, feat3
0.1, 0.2, 0.5
-0.4, 0.2, 0.5

the column passed as target is used as the y-vector and the remaining columns are used as the X
matrix.

The coefficient vector is extracted from a csv file of the form (mycoeff.json in the example below)

  {
	"TargetName": "feat3",
	"Datafile": "gafit/_testdata/dataset.csv",
	"Coeffs": {
	  "Var1": 2.9999990000004804,
	  "Var2": 1.0000003999997198
	},
	"Score": {
	  "Name": "aicc",
	  "Value": -25.7622420881808
	}
  }

where the first column is a name (that must match one of header fields in the data csvfile) and
the second column is the value of the coefficients. Coefficients corresponding to columns in the
data matrix that are not listed, is taken as zero.

Minimal example:

gogafit rmse -d mydata.csv -c mycoeff,csv

Usage:
  gogafit rmse [flags]

Flags:
  -d, --data string    Csv file with data
  -h, --help           help for rmse
  -m, --model string   JSON file with fitted model coefficients (default "model.json")

Global Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
```
## Plot command
```
Create a scatter plot of the predictions of one or multiple datasets.
If we have the training data in a file called train.csv and validation data in a file
validate.csv. Our trained model is stored in model.json, it can be plotted by

gogafit plot -d train.csv,validate.csv -m model.json -o plot.png

Usage:
  gogafit plot [flags]

Flags:
  -d, --data string    Comma separated list of datasets (e.g. test, train
  -h, --help           help for plot
  -m, --model string   JSON file with the model
  -o, --out string     Image file where the model will be stored (default "gogafitPlot.png")

Global Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
```
## Poly command
```
Adds columns representing a polynomial version of a subset of the columns.
	
Example:
We have the following csv file named data.csv

feat1,feat2,targetQuantity
1.0,2.0,3.0
2.0,1.0,2.0

and we wich to add a new feature which is feat1^2. Run

gogafit poly -d data.csv -t targetQuantity -o 2 -p feat1

this will create a file data_poly.csv

feat1,feat2,feat1p2targetQuantity
1.0,2.0,1.0,3.0
2.0,1.0,4.0,2.0

Usage:
  gogafit poly [flags]

Flags:
  -d, --data string      Original datafile
  -h, --help             help for poly
  -o, --order uint       Polynomial order (default 1)
  -p, --pattern string   Polynomial versions of all features containing this substring will be added
  -y, --target string    Name of the quantity used as target property

Global Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
```
## ELM command
```
Calculates the output of a hiddan layer in an extreme learning machine.
The hidden layer consists of the given number of activation functions.

gogafit elm -d mydata.csv -r 100 -s 200 -t feat3 

creates an ELM with 100 relu neurons and 200 sigmoid neurons. Similar to the other commands,
the format of the data file is

feat1, feat2, feat3
0.2, 0.1, 0.4
...

where the name of the column corresponding to the target feature is specified via the -y flag.

Usage:
  gogafit elm [flags]

Flags:
  -d, --data string     Datafile with inputs for the input layer
  -h, --help            help for elm
  -r, --relu uint       Number of rectifier activation functions in the hidden layer (default 1)
  -s, --sig uint        Number of sigmoid activation functions in the hidden layer (default 1)
  -y, --target string   Name of columns that represent the target values

Global Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
```
## Hook command
```
This command generates templates for hooks

Usage:
  gogafit hook [flags]

Flags:
  -h, --help            help for hook
  -o, --out string      output file (default "cost.py")
  -p, --pyexec string   name of the python executable (default "python")
  -t, --type string     Template type (currently only cost supported) (default "cost")

Global Flags:
      --config string   config file (default is $HOME/.gogafit.yaml)
```
