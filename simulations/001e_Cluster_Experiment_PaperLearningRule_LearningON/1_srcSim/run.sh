startTime=$(date +%s)

let parallel=9
let durchgaenge=2

for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python parallel.py --simID $y &
        done
        wait
        sleep 5
	rm -r *annarchy*
	endTime=$(date +%s)
	dif=$((endTime - startdurchgangTime))
    echo "${durchgang} ${dif}" >> fertig_zeit.txt
	sleep 5
done

endTime=$(date +%s)
dif=$((endTime - startTime))
echo $dif >> fertig_zeit.txt

