OUTFILE=docs/scripts.md 
rm -f $OUTFILE
printf "# Script overview\n" >> $OUTFILE
for file in *.py
do
    printf "## $file\n" >> $OUTFILE
    python3 $file --help >> $OUTFILE
    printf "\n" >> $OUTFILE
done
    