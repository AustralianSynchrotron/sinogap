#!/bin/bash


if [ -z "${1}" ] ; then
  echo "No input mask (first argument) was given." >&2
  printhelp >&2
  exit 1
fi
iMask="${1}"

if [ -z "${2}" ] ; then
  echo "No output destination (second argument) was given." >&2
  printhelp >&2
  exit 1
fi
oDir="${2}"

mkdir -p "$oDir"
ls sinogap_module*.py *${iMask}* | 
while read flnm ; do
  onm=$(sed "s:${iMask}::g" <<< "${flnm}" )
  cp -Lvr "$flnm" "$oDir/$onm"
done
#echo cp -r sinogap_module.py *${iMask}* "$oDir"
#cp -rf runs/*${iMask}/* "$oDir/logs"
rsync -av runs/*${iMask}/ "$oDir/logs"
 


