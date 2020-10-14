package cmd

import (
	"log"
	"os"

	"github.com/davidkleiven/gogafit/gafit"
	"github.com/spf13/cobra"
)

// hookCmd represents the hook command
var hookCmd = &cobra.Command{
	Use:   "hook",
	Short: "Generate templates scripts for hooks",
	Long:  `This command generates templates for hooks`,
	Run: func(cmd *cobra.Command, args []string) {
		templateType, err := cmd.Flags().GetString("type")
		if err != nil {
			log.Fatalf("Error %s\n", err)
			return
		}

		out, err := cmd.Flags().GetString("out")
		if err != nil {
			log.Fatalf("Error %s\n", err)
			return
		}

		pyexec, err := cmd.Flags().GetString("pyexec")
		if err != nil {
			log.Fatalf("Error %s\n", err)
			return
		}

		switch templateType {
		case "cost":
			script, err := gafit.DemoCostFuncPython(pyexec)
			if err != nil {
				log.Fatalf("%s\n", err)
				return
			}
			if err = saveScript(out, script); err != nil {
				log.Fatalf("%s\n", err)
			}
			log.Printf("Script written to %s\n", out)
			break
		default:
			log.Printf("Unknown script template %s\n", templateType)
		}

	},
}

func saveScript(fname string, script string) error {
	f, err := os.OpenFile(fname, os.O_CREATE|os.O_WRONLY, 0755)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(script)
	return err
}

func init() {
	rootCmd.AddCommand(hookCmd)

	hookCmd.Flags().StringP("type", "t", "cost", "Template type (currently only cost supported)")
	hookCmd.Flags().StringP("out", "o", "cost.py", "output file")
	hookCmd.Flags().StringP("pyexec", "p", "python", "name of the python executable")
}
