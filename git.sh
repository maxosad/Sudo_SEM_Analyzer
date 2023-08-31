#!/bin/bash
git add .
echo "write commit message"
read message
git commit -m "$message"
git push