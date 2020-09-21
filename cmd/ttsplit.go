package cmd

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/spf13/cobra"
)

// ttsplitCmd represents the ttsplit command
var ttsplitCmd = &cobra.Command{
	Use:   "ttsplit",
	Short: "Split a dataset in a train and test set",
	Long: `Splits the passed csv file into a training set and a validation set.
If the data file is called mydata.csv, the program will create two file

mydata_train.csv for the training data
mydate_test.csv for the test/validation data
	`,
	Run: func(cmd *cobra.Command, args []string) {
		dataFile, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		frac, err := cmd.Flags().GetFloat64("fraction")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		f, err := os.Open(dataFile)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		lines := []string{}
		for scanner.Scan() {
			lines = append(lines, scanner.Text())
		}

		header := lines[0]
		lines = lines[1:]
		rand.Shuffle(len(lines), func(i, j int) { lines[i], lines[j] = lines[j], lines[i] })

		num := int(frac * float64(len(lines)))
		test := lines[:num]
		train := lines[num:]

		err = write(outfname(dataFile, "_test"), header, test)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		err = write(outfname(dataFile, "_train"), header, train)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
	},
}

func outfname(fname string, identifier string) string {
	prefix := fname[:len(fname)-4]
	return fmt.Sprintf("%s%s.csv", prefix, identifier)
}

func write(fname string, header string, lines []string) error {
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer f.Close()

	f.WriteString(header + "\n")
	for _, l := range lines {
		f.WriteString(l + "\n")
	}
	return nil
}

func init() {
	rootCmd.AddCommand(ttsplitCmd)

	ttsplitCmd.Flags().StringP("data", "d", "", "Dataset with data")
	ttsplitCmd.Flags().Float64P("fraction", "f", 0.2, "Fraction of the data placed in the test set")
}
