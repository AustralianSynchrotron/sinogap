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

mkdir -p "$oDir/runs/"
cp -r sinogap*.py *${iMask}* "$oDir"
cp -r runs/*${iMask}* "$oDir/runs/"
 


