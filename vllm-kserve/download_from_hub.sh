#!bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -m <MODEL_NAME>"
   exit 1 # Exit script after printing help
}

while getopts ":m:" opt;
do
   case "$opt" in
        m ) repo_id="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

function make_download_cmd()
{
    if [ ! -z $repo_id ] ; then
        cmd="huggingface-cli download $repo_id"
     else
        echo "Model not provided"
        helpFunction
    fi

    cmd+=" --local-dir models/$repo_id --cache-dir models/$repo_id --exclude *.pt"
}

make_download_cmd
echo ""
echo "Running: $cmd"
echo ""
$cmd
