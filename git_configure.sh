#!/usr/bin/env sh

	git config filter.strip-notebook-output.clean 'jupyter nbconvert --clear-output --to=notebook --stdin --stdout --log-level=INFO'

	mkdir notebook

	cat << EOT > notebook/.gitattributes
	*.ipynb filter=strip-notebook-output
	EOT

	git add notebook/.gitattributes
	