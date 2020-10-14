FILE="cliHelp.md"

echo "# CLI Commands" > $FILE

echo "\`\`\`" >> $FILE
go run main.go -h >> $FILE
echo "\`\`\`" >> $FILE

echo "## Fit command" >> $FILE
echo "\`\`\`" >> $FILE
go run main.go fit -h >> $FILE
echo "\`\`\`" >> $FILE

echo "## TTsplit command" >> $FILE
echo "\`\`\`" >> $FILE
go run main.go ttsplit -h >> $FILE
echo "\`\`\`" >> $FILE

echo "## RMSE command" >> $FILE
echo "\`\`\`" >> $FILE
go run main.go rmse -h >> $FILE
echo "\`\`\`" >> $FILE

echo "## Plot command" >> $FILE
echo "\`\`\`" >> $FILE
go run main.go plot -h >> $FILE
echo "\`\`\`" >> $FILE

echo "## Poly command" >> $FILE
echo "\`\`\`" >> $FILE
go run main.go poly -h >> $FILE
echo "\`\`\`" >> $FILE

echo "## ELM command" >> $FILE
echo "\`\`\`" >> $FILE
go run main.go elm -h >> $FILE
echo "\`\`\`" >> $FILE

echo "## Hook command" >> $FILE
echo "\`\`\`" >> $FILE
go run main.go hook -h >> $FILE
echo "\`\`\`" >> $FILE 
