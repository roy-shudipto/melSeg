# flake8
max_line_length=100
flake8 --max-line-length=$max_line_length ./*.py
flake8 --max-line-length=$max_line_length ./*/*.py

# black
black ./*.py --preview
black ./*/*.py --preview