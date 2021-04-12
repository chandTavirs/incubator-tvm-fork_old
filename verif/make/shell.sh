################################################################################
# Makefile shell functions.
################################################################################
# Modified by contributors from Intel Labs

red="\x1b[31m"
grn="\x1b[32m"
ylw="\x1b[33m"
blu="\x1b[34m"
end="\x1b(B\x1b[m"

color () { 
  sed -e "s/PASS/${grn}PASS${end}/g" \
      -e "s/TRUE/${grn}TRUE${end}/g" \
      -e "s/DONE/${grn}DONE${end}/g" \
      -e "s/FAIL/${red}FAIL${end}/g" \
      -e "s/FALSE/${red}FALSE${end}/g" \
      -e "s/TODO/${ylw}TODO${end}/g" \
      -e "s/MAKE/${blu}MAKE${end}/g" \
      -e "s/NONE/${ylw}NONE${end}/g"
}

run_task () {
  job="$1"
  cmd="$2"
  stem="${job%.*}"
  stat="$stem.stat"
  echo "$cmd" > $job
  date="`date +'%F %T'`"
  start=$SECONDS
  status="MAKE"
  printf "%-48s $status on $date\n" "$stem" | tee $stat | color
  rm -f $stem.fail
  if eval $cmd >& $job; then
    status="PASS"
    ret_status=0
  else
    status="FAIL"
    ret_status=1
    mv -f $job $stem.fail
  fi
  elapsed=$((SECONDS-start))
  hor="$((elapsed/3600))"
  min="$((($elapsed%3600)/60))"
  sec="$(($elapsed%60))"
  date="`date +'%F %T'`"
  elapsed="$hor:$min:$sec"
  printf "%-48s $status on $date in %02d:%02d:%02d\n" \
         "$stem" "$hor" "$min" "$sec" | tee $stat | color
  #return $ret_status
}

"$@"
